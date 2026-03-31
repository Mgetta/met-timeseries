"""
Source-agnostic derived variable calculations.

Each function is a pure transformation on :class:`xarray.DataArray` inputs —
no knowledge of NLDAS/PRISM variable names or :class:`xarray.Dataset` structure.
The caller maps source-specific variable names to function arguments.

Derivations happen on the grid **before** spatial aggregation because nonlinear
functions do not commute with spatial averaging:
``mean(sqrt(u² + v²)) ≠ sqrt(mean(u)² + mean(v)²)``.
"""

from __future__ import annotations

import datetime

import numpy as np
import xarray as xr


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Compute wind speed from U and V components using MetPy.

    Parameters
    ----------
    u:
        U-component of wind in m/s.
    v:
        V-component of wind in m/s.

    Returns
    -------
    :class:`xarray.DataArray` named ``wind_speed_ms``.
    """
    from metpy.calc import wind_speed as _mpc_wind_speed
    from metpy.units import units

    u_q = u.values * units("m/s")
    v_q = v.values * units("m/s")
    ws = _mpc_wind_speed(u_q, v_q)
    return xr.DataArray(
        ws.magnitude,
        dims=u.dims,
        coords=u.coords,
        name="wind_speed_ms",
    )


def dewpoint_from_specific_humidity(
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute dewpoint temperature from specific humidity.

    Uses ``metpy.calc.dewpoint_from_specific_humidity`` for the full
    thermodynamic chain: specific humidity → mixing ratio → vapour pressure →
    dewpoint, which is more accurate than a simple Magnus approximation.

    Parameters
    ----------
    specific_humidity:
        Specific humidity in kg/kg.
    pressure:
        Atmospheric pressure in Pa.  Defaults to standard atmosphere
        (101 325 Pa) when not provided.

    Returns
    -------
    :class:`xarray.DataArray` named ``dewpoint_c`` in degrees Celsius.
    """
    from metpy.calc import dewpoint_from_specific_humidity as _mpc_dp
    from metpy.units import units

    if pressure is None:
        pres_pa = np.full_like(specific_humidity.values, 101325.0) * units("Pa")
    else:
        pres_pa = pressure.values * units("Pa")

    spfh = specific_humidity.values * units("kg/kg")
    dp = _mpc_dp(pres_pa, spfh)

    return xr.DataArray(
        dp.to("degC").magnitude,
        dims=specific_humidity.dims,
        coords=specific_humidity.coords,
        name="dewpoint_c",
    )

def kelvin_to_fahrenheit(temp_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Fahrenheit.

    Parameters
    ----------
    temp_k:
        Temperature in Kelvin.

    Returns
    -------
    :class:`xarray.DataArray` named ``temp_f`` in degrees Fahrenheit.
    """
    return ((temp_k - 273.15) * 9/5 + 32).rename("temp_f")


def kelvin_to_celsius(temp_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    temp_k:
        Temperature in Kelvin.

    Returns
    -------
    :class:`xarray.DataArray` named ``temp_c`` in degrees Celsius.
    """
    return (temp_k - 273.15).rename("temp_c")


def cloud_cover(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray:
    """Estimate cloud-cover fraction from observed shortwave and clear-sky radiation.

    ``cloud_cover = 1 - observed / clear_sky``, clamped to ``[0, 1]``.
    Pixels where the theoretical clear-sky radiation is below 10 W/m²
    (nighttime / near-horizon) are set to NaN.

    Parameters
    ----------
    shortwave:
        Observed shortwave radiation in W/m².  Expected dims are
        ``(time, lat, lon)`` when *time* is provided, or ``(lat, lon)``
        otherwise.
    lat:
        Latitude values in decimal degrees.
    lon:
        Longitude values in decimal degrees.
    time:
        Array of datetime-like values used for solar-geometry calculations.
        When ``None`` a fixed reference time (2000-07-01 12:00 UTC) is used
        so results are reproducible.

    Returns
    -------
    :class:`xarray.DataArray` named ``cloud_cover_fraction``.
    """
    lat_vals = np.asarray(lat)
    lon_vals = np.asarray(lon)

    if time is not None:
        time_values = np.asarray(time)
        clearsky_values = np.full(shortwave.shape, np.nan, dtype=float)

        time_axis = shortwave.dims.index("time")
        lat_axis = shortwave.dims.index("lat") if "lat" in shortwave.dims else None
        lon_axis = shortwave.dims.index("lon") if "lon" in shortwave.dims else None

        for t_idx, t_val in enumerate(time_values):
            dt = _to_datetime(t_val)
            cs_slice = np.array(
                [[_clear_sky_radiation(la, lo, dt) for lo in lon_vals] for la in lat_vals],
                dtype=float,
            )
            idx: list = [slice(None)] * clearsky_values.ndim
            idx[time_axis] = t_idx
            if lat_axis is not None and lon_axis is not None:
                adjusted_lat = lat_axis if lat_axis < time_axis else lat_axis - 1
                adjusted_lon = lon_axis if lon_axis < time_axis else lon_axis - 1
                cs_reordered = np.moveaxis(
                    cs_slice,
                    [0, 1],
                    [adjusted_lat, adjusted_lon],
                )
                clearsky_values[tuple(idx)] = cs_reordered
            else:
                clearsky_values[tuple(idx)] = cs_slice

        clearsky = xr.DataArray(clearsky_values, dims=shortwave.dims, coords=shortwave.coords)
    else:
        dt = datetime.datetime(2000, 7, 1, 12, 0, 0)
        cs_values = np.array(
            [[_clear_sky_radiation(la, lo, dt) for lo in lon_vals] for la in lat_vals],
            dtype=float,
        )
        lat_dim = "lat" if "lat" in shortwave.dims else shortwave.dims[0]
        lon_dim = "lon" if "lon" in shortwave.dims else shortwave.dims[1]
        clearsky = xr.DataArray(
            cs_values,
            dims=[lat_dim, lon_dim],
            coords={
                lat_dim: shortwave.coords.get(lat_dim),
                lon_dim: shortwave.coords.get(lon_dim),
            },
        )

    _MIN_CLEARSKY = 10.0
    daytime_mask = clearsky > _MIN_CLEARSKY

    with np.errstate(divide="ignore", invalid="ignore"):
        cc = 1.0 - shortwave / clearsky

    cc = cc.clip(0.0, 1.0)
    cc = cc.where(daytime_mask)
    return cc.rename("cloud_cover_fraction")


# ---------------------------------------------------------------------------
# PET
# ---------------------------------------------------------------------------


def pet_hargreaves(
    daily_tmin,
    daily_tmax,
    lat: float,
):
    """Estimate daily PET using the Hargreaves (1985) method.

    Uses ``mettoolbox.utils.radiation`` for extraterrestrial radiation (Ra)
    computed from latitude and day-of-year.

    Formula:
        ``PET = 0.0023 × Ra × (T_mean + 17.8) × sqrt(T_max - T_min)``

    where Ra is in MJ/m²/day and PET is in mm/day.

    Parameters
    ----------
    daily_tmin:
        Daily minimum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    daily_tmax:
        Daily maximum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    lat:
        Latitude in decimal degrees.

    Returns
    -------
    :class:`pandas.Series` named ``pet_hargreaves_mm``.
    """
    import pandas as pd
    from mettoolbox.utils import radiation

    tmean = (daily_tmin + daily_tmax) / 2.0
    trange = (daily_tmax - daily_tmin).clip(lower=0.0)

    # mettoolbox.utils.radiation expects a Series/DataFrame with a DatetimeIndex
    ra_df = radiation(tmean.to_frame(name="temp"), lat)
    ra = ra_df["ra"]

    with np.errstate(invalid="ignore"):
        pet = 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange)

    pet = pet.fillna(0.0).clip(lower=0.0)
    pet.name = "pet_hargreaves_mm"
    return pet


def pet_penman_monteith(
    daily_tmin: xr.DataArray,
    daily_tmax: xr.DataArray,
    wind_speed: xr.DataArray,
    shortwave: xr.DataArray,
    dewpoint: xr.DataArray,
    elevation: "xr.DataArray | float",
) -> xr.DataArray:
    """Compute FAO-56 Penman-Monteith reference ET at the grid scale.

    Implements the full FAO-56 reference evapotranspiration equation
    (FAO Irrigation and Drainage Paper 56, Chapter 3):

    .. math::

        ET_0 = \\frac{0.408 \\Delta (R_n - G) + \\gamma
                      \\frac{900}{T+273} u_2 (e_s - e_a)}
                     {\\Delta + \\gamma (1 + 0.34 u_2)}

    All intermediate calculations follow FAO-56 exactly.  The function
    operates on raw gridded ``xr.DataArray`` inputs **before** spatial
    aggregation so that nonlinear operations (Magnus exponential,
    Stefan-Boltzmann T⁴, etc.) are applied per grid cell.

    Parameters
    ----------
    daily_tmin:
        Daily minimum air temperature in °C, dims ``(time, lat, lon)``.
    daily_tmax:
        Daily maximum air temperature in °C, dims ``(time, lat, lon)``.
    wind_speed:
        Wind speed at 2 m height in m/s, dims ``(time, lat, lon)``.
    shortwave:
        Incoming shortwave radiation in MJ/m²/day, dims ``(time, lat, lon)``.
    dewpoint:
        Dewpoint temperature in °C (used for actual vapour pressure),
        dims ``(time, lat, lon)``.
    elevation:
        Elevation above sea level in metres.  May be a scalar ``float`` or
        a gridded :class:`~xarray.DataArray` broadcastable to the spatial
        dimensions of the other inputs.

    Returns
    -------
    :class:`xarray.DataArray` named ``pet_penman_monteith_mm`` in mm/day,
    with the same dims and coords as the input DataArrays.  Values are
    clipped to ``≥ 0``; NaN cells are filled with ``0.0``.
    """
    # -- raw numpy arrays for arithmetic (consistent with other derivations) --
    tmin = daily_tmin.values.astype(float)
    tmax = daily_tmax.values.astype(float)
    u2 = wind_speed.values.astype(float)
    rs = shortwave.values.astype(float)
    td = dewpoint.values.astype(float)

    if isinstance(elevation, xr.DataArray):
        z = elevation.values.astype(float)
    else:
        z = float(elevation)

    tmean = (tmin + tmax) / 2.0

    # -- Step 1: Atmospheric pressure (FAO-56 Eq. 7) --
    pressure = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26  # kPa

    # -- Step 2: Psychrometric constant γ (FAO-56 Eq. 8) --
    gamma = 0.000665 * pressure  # kPa/°C

    # -- Step 3: Saturation vapour pressure eₛ (FAO-56 Eq. 11, 12) --
    def _magnus(t):
        return 0.6108 * np.exp(17.27 * t / (t + 237.3))

    es = (_magnus(tmax) + _magnus(tmin)) / 2.0  # kPa

    # -- Step 4: Actual vapour pressure eₐ from dewpoint (FAO-56 Eq. 14) --
    ea = _magnus(td)  # kPa

    # -- Step 5: Slope of saturation VP curve Δ (FAO-56 Eq. 13) --
    delta = 4098.0 * _magnus(tmean) / (tmean + 237.3) ** 2  # kPa/°C

    # -- Step 6: Extraterrestrial radiation Ra (FAO-56 Eq. 21-25) --
    # Latitude comes from the lat coordinate of the input DataArrays.
    lat_coord = daily_tmin.coords["lat"].values  # degrees
    lat_rad = np.radians(lat_coord)  # (n_lat,)

    # Day of year from the time coordinate
    import pandas as pd

    time_values = daily_tmin.coords["time"].values
    doy = np.array(
        [pd.Timestamp(t).timetuple().tm_yday for t in time_values], dtype=float
    )  # (n_time,)

    # Solar declination (rad) and inverse relative distance Earth-Sun
    # FAO-56 Eq. 24, 23
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * doy)  # (n_time,)
    dec = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)  # (n_time,)

    # Broadcast to (time, lat, lon) for vectorised computation.
    # Determine axis positions from the dims of the input array.
    time_ax = daily_tmin.dims.index("time")
    lat_ax = daily_tmin.dims.index("lat")
    lon_ax = daily_tmin.dims.index("lon")
    ndim = len(daily_tmin.dims)

    def _expand(arr, axis, total_dims):
        """Insert length-1 axes so *arr* broadcasts to *total_dims* dimensions."""
        shape = [1] * total_dims
        shape[axis] = len(arr)
        return arr.reshape(shape)

    dr_b = _expand(dr, time_ax, ndim)
    dec_b = _expand(dec, time_ax, ndim)
    lat_b = _expand(lat_rad, lat_ax, ndim)

    # Sunset hour angle ωs (FAO-56 Eq. 25) — clamp argument to [-1, 1] for
    # polar latitudes where sun is permanently above/below horizon.
    ws_arg = np.clip(-np.tan(lat_b) * np.tan(dec_b), -1.0, 1.0)
    ws = np.arccos(ws_arg)  # (time, lat, 1) broadcastable

    # Ra in MJ/m²/day (FAO-56 Eq. 21)
    gsc = 0.0820  # solar constant MJ/m²/min
    ra = (
        24.0
        * 60.0
        / np.pi
        * gsc
        * dr_b
        * (
            ws * np.sin(lat_b) * np.sin(dec_b)
            + np.cos(lat_b) * np.cos(dec_b) * np.sin(ws)
        )
    )
    ra = np.broadcast_to(ra, tmin.shape).copy()  # ensure writeable

    # -- Step 7: Clear-sky radiation Rso (FAO-56 Eq. 37) --
    rso = (0.75 + 2e-5 * z) * ra  # MJ/m²/day

    # -- Step 8: Net shortwave radiation Rns (FAO-56 Eq. 38) --
    albedo = 0.23
    rns = (1.0 - albedo) * rs  # MJ/m²/day

    # -- Step 9: Net longwave radiation Rnl (FAO-56 Eq. 39) --
    sigma = 4.903e-9  # Stefan-Boltzmann MJ/(m²·day·K⁴)
    tmax_k = tmax + 273.15
    tmin_k = tmin + 273.15
    t_term = sigma * (tmax_k**4 + tmin_k**4) / 2.0
    humidity_term = 0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0))

    # Cloudiness correction — clip Rs/Rso to [0.25, 1.0] per FAO-56 guidance.
    # Coefficients 1.35 and 0.35 are empirical (FAO-56 Eq. 39).
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_rso_ratio = np.where(rso > 0.0, rs / rso, 0.25)
    rs_rso_ratio = np.clip(rs_rso_ratio, 0.25, 1.0)
    cloud_term = 1.35 * rs_rso_ratio - 0.35

    rnl = t_term * humidity_term * cloud_term  # MJ/m²/day

    # -- Step 10: Net radiation Rn (FAO-56 Eq. 40) --
    rn = rns - rnl  # MJ/m²/day

    # -- Step 11: Soil heat flux G = 0 for daily timestep --
    g = 0.0

    # -- Step 12: FAO-56 Penman-Monteith equation --
    numerator = 0.408 * delta * (rn - g) + gamma * (900.0 / (tmean + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)

    with np.errstate(divide="ignore", invalid="ignore"):
        et0 = numerator / denominator

    et0 = np.where(np.isfinite(et0), et0, 0.0)
    et0 = np.clip(et0, 0.0, None)

    return xr.DataArray(
        et0,
        dims=daily_tmin.dims,
        coords=daily_tmin.coords,
        name="pet_penman_monteith_mm",
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clear_sky_radiation(lat: float, lon: float, dt: datetime.datetime) -> float:
    """Compute theoretical clear-sky surface shortwave radiation (W/m²).

    Uses standard solar geometry with a fixed atmospheric transmittance.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees.
    lon:
        Longitude in decimal degrees (retained for future equation-of-time
        corrections).
    dt:
        UTC datetime for which to compute the radiation.

    Returns
    -------
    Clear-sky surface shortwave radiation in W/m², or 0.0 at nighttime.
    """
    S0 = 1361.0  # solar constant (W/m²)
    tau = 0.75   # clear-sky atmospheric transmittance

    n = dt.timetuple().tm_yday  # day of year

    dec_deg = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + n)))
    dec_rad = np.radians(dec_deg)

    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    omega_rad = np.radians(15.0 * (hour - 12.0))

    lat_rad = np.radians(lat)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(omega_rad)
    )

    if cos_zenith <= 0.0:
        return 0.0

    return float(S0 * cos_zenith * tau)


def _to_datetime(t_val) -> datetime.datetime:
    """Convert an arbitrary numpy/pandas timestamp to a :class:`datetime.datetime`."""
    import pandas as pd

    return pd.Timestamp(t_val).to_pydatetime(warn=False)
