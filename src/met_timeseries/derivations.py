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


def adjust_wind_height(
    wind_speed: xr.DataArray,
    z_from: float = 10.0,
    z_to: float = 2.0,
    method: str = "logarithmic",
    alpha: float = 1 / 7,
) -> xr.DataArray:
    """Adjust wind speed between measurement heights.

    Two methods are available:

    **Logarithmic** (FAO-56 Eq. 47):

    .. math::

        u_{z_{to}} = u_{z_{from}} \\times
        \\frac{\\ln(67.8\\, z_{to} - 5.42)}{\\ln(67.8\\, z_{from} - 5.42)}

    Derived from Monin-Obukhov similarity theory under neutral stability,
    with an implicit roughness length of ~0.01 m (short grass).
    Reference: Allen et al. (1998), FAO Irrigation & Drainage Paper 56.

    **Power law** (Hellman / Davenport):

    .. math::

        u_{z_{to}} = u_{z_{from}} \\times
        \\left(\\frac{z_{to}}{z_{from}}\\right)^{\\alpha}

    where α is a surface-roughness exponent.  Common values:

    =============================  =====
    Surface type                   α
    =============================  =====
    Open water / ice               0.10
    Open flat terrain (1/7 rule)   0.143
    Agricultural crops             0.20
    Suburban / forest              0.25
    Urban                          0.35
    =============================  =====

    Reference: Davenport (1960); Justus & Mikhail (1976).

    Parameters
    ----------
    wind_speed:
        Wind speed in m/s at height ``z_from``.
    z_from:
        Source measurement height in metres. Default is 10 m (NLDAS).
    z_to:
        Target height in metres. Default is 2 m (FAO-56 standard).
    method:
        ``"logarithmic"`` (FAO-56) or ``"power_law"`` (Hellman).
    alpha:
        Power-law exponent. Only used when ``method="power_law"``.
        Default is 1/7 ≈ 0.143 (open flat terrain).

    Returns
    -------
    :class:`xarray.DataArray` named ``wind_speed_ms`` adjusted to ``z_to``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"logarithmic"`` or ``"power_law"``.
    """
    if method == "logarithmic":
        # FAO-56 Eq. 47: u_2 = u_z × 4.87 / ln(67.8z - 5.42)
        # Generalised for arbitrary z_to:
        #   u_z_to = u_z_from × ln(67.8·z_to - 5.42) / ln(67.8·z_from - 5.42)
        factor = np.log(67.8 * z_to - 5.42) / np.log(67.8 * z_from - 5.42)
        adjusted = wind_speed * factor

    elif method == "power_law":
        # Hellman / Davenport: u_z_to = u_z_from × (z_to / z_from)^α
        adjusted = wind_speed * (z_to / z_from) ** alpha

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'logarithmic' or 'power_law'."
        )

    return adjusted.rename("wind_speed_ms")



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

def dewpoint_from_specific_humidity_cc(
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Compute dewpoint from specific humidity using Clausius-Clapeyron.

    Replicates the methodology from the original MetTool:

    1. Specific humidity → mixing ratio: ``w = q / (1 − q)``
    2. Mixing ratio → actual vapor pressure: ``eₐ = w / (0.622 + w) × P``
    3. Clausius-Clapeyron inversion:
       ``Td = 1 / (1/273.15 − 0.0001844 × ln(eₐ / 0.6113))``

    The constant ``0.0001844`` derives from ``Rᵥ / Lᵥ`` where
    ``Rᵥ = 461.5 J/(kg·K)`` (specific gas constant for water vapor) and
    ``Lᵥ ≈ 2.501e6 J/kg`` (latent heat of vaporization at 0°C):
    ``Rᵥ / Lᵥ = 461.5 / 2501000 ≈ 0.0001844``.

    The reference vapor pressure ``0.6113 kPa`` is the saturation vapor
    pressure at 0°C (273.15 K) from the Clausius-Clapeyron relation.

    Parameters
    ----------
    specific_humidity:
        Specific humidity in kg/kg.
    pressure:
        Atmospheric pressure in Pa.

    Returns
    -------
    :class:`xarray.DataArray` named ``dewpoint_c`` in degrees Celsius.
    """
    q = specific_humidity.values.astype(float)
    p_kpa = pressure.values.astype(float) / 1000.0  # Pa → kPa

    # Step 1: specific humidity → mixing ratio
    w = q / (1.0 - q)

    # Step 2: mixing ratio → actual vapor pressure (kPa)
    e_a = (w / (0.622 + w)) * p_kpa

    # Step 3: Clausius-Clapeyron inversion → dewpoint (K)
    REF_VP = 0.6113  # saturation VP at 0°C (kPa)
    CC_CONST = 0.0001844  # Rv / Lv (K⁻¹)

    with np.errstate(divide="ignore", invalid="ignore"):
        td_k = 1.0 / (1.0 / 273.15 - CC_CONST * np.log(e_a / REF_VP))

    # K → °C
    td_c = td_k - 273.15

    return xr.DataArray(
        td_c,
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


def pet_penman_monteith_hourly(
    temperature: xr.DataArray,
    wind_speed: xr.DataArray,
    shortwave: xr.DataArray,
    dewpoint: xr.DataArray,
    elevation: "xr.DataArray | float",
) -> xr.DataArray:
    """Compute hourly FAO-56 Penman-Monteith reference ET at the grid scale.

    Implements the hourly FAO-56 reference evapotranspiration equation
    (FAO Irrigation and Drainage Paper 56, Chapter 4, Eq. 53):

    .. math::

        ET_0 = \\frac{0.408 \\Delta (R_n - G) + \\gamma
                      \\frac{37}{T+273} u_2 (e^\\circ(T) - e_a)}
                     {\\Delta + \\gamma (1 + C_d u_2)}

    where :math:`C_d = 0.24` during daytime and :math:`0.96` at night.

    All intermediate calculations follow FAO-56 Chapter 4 exactly.  The
    function operates on raw gridded ``xr.DataArray`` inputs **before**
    spatial aggregation so that nonlinear operations are applied per grid
    cell.

    Parameters
    ----------
    temperature:
        Instantaneous air temperature in °C, dims ``(time, lat, lon)``.
    wind_speed:
        Wind speed at 2 m height in m/s, dims ``(time, lat, lon)``.
    shortwave:
        Incoming shortwave radiation in **W/m²** (converted internally to
        MJ/m²/hour via ``× 0.0036``), dims ``(time, lat, lon)``.
    dewpoint:
        Dewpoint temperature in °C (used for actual vapour pressure),
        dims ``(time, lat, lon)``.
    elevation:
        Elevation above sea level in metres.  May be a scalar ``float`` or
        a gridded :class:`~xarray.DataArray` broadcastable to the spatial
        dimensions of the other inputs.

    Returns
    -------
    :class:`xarray.DataArray` named ``pet_penman_monteith_hourly_mm`` in
    mm/hour, with the same dims and coords as the input DataArrays.  Values
    are clipped to ``≥ 0``; NaN cells are filled with ``0.0``.

    Notes
    -----
    Shortwave radiation is accepted in W/m² (native NLDAS units) and
    converted to MJ/m²/hour internally (multiply by 0.0036).

    Key differences from the daily version (FAO-56 Chapter 3):

    * Temperature: instantaneous T at each hour (not Tmin/Tmax average).
    * Saturation VP: single Magnus at hourly T, not (e°(Tmax)+e°(Tmin))/2.
    * Radiation: MJ/m²/hour (not MJ/m²/day).
    * Ra: hourly integral using ω₁, ω₂ hour angles (FAO-56 Eq. 28).
    * Soil heat flux: G = 0.1·Rn daytime, G = 0.5·Rn nighttime.
    * Wind coefficient Cn = 37 (not 900).
    * Resistance coefficient Cd = 0.24 daytime, 0.96 nighttime.
    * Stefan-Boltzmann in MJ/(m²·hour·K⁴) = 2.042e-10.
    """
    import pandas as pd

    # -- Empirical constants (FAO-56 Chapter 4) --
    cn = 37.0           # wind function numerator coefficient (hourly)
    cd_day = 0.24       # wind function denominator coefficient (daytime)
    cd_night = 0.96     # wind function denominator coefficient (nighttime)
    g_frac_day = 0.1    # G = g_frac_day * Rn during daytime
    g_frac_night = 0.5  # G = g_frac_night * Rn during nighttime
    albedo = 0.23       # reference crop (grass) albedo (FAO-56 Eq. 38)
    # Cloudiness correction coefficients (FAO-56 Eq. 39, hourly form)
    cloud_a = 1.35
    cloud_b = 0.35
    # Humidity coefficient for net longwave (FAO-56 Eq. 39)
    humid_a = 0.34
    humid_b = 0.14

    # -- raw numpy arrays --
    t = temperature.values.astype(float)
    u2 = wind_speed.values.astype(float)
    rs_wm2 = shortwave.values.astype(float)
    td = dewpoint.values.astype(float)

    if isinstance(elevation, xr.DataArray):
        z = elevation.values.astype(float)
    else:
        z = float(elevation)

    # Convert shortwave from W/m² to MJ/m²/hour
    rs = rs_wm2 * 0.0036

    # -- Step 1: Atmospheric pressure (FAO-56 Eq. 7) --
    pressure = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26  # kPa

    # -- Step 2: Psychrometric constant γ (FAO-56 Eq. 8) --
    gamma = 0.000665 * pressure  # kPa/°C

    # -- Step 3: Magnus formula helper --
    def _magnus(temp):
        return 0.6108 * np.exp(17.27 * temp / (temp + 237.3))

    # -- Step 4: Saturation VP at hourly T (FAO-56 Eq. 11) --
    # For hourly, eₛ = e°(T) directly (not average of Tmax/Tmin)
    es = _magnus(t)  # kPa

    # -- Step 5: Actual VP from dewpoint (FAO-56 Eq. 14) --
    ea = _magnus(td)  # kPa

    # -- Step 6: Slope of saturation VP curve Δ (FAO-56 Eq. 13) --
    delta = 4098.0 * es / (t + 237.3) ** 2  # kPa/°C

    # -- Step 7: Extraterrestrial radiation Ra (FAO-56 Eq. 28) --
    # Latitude from the DataArray lat coordinate
    lat_coord = temperature.coords["lat"].values  # degrees
    lat_rad = np.radians(lat_coord)  # (n_lat,)

    # Longitude from the DataArray lon coordinate
    lon_coord = temperature.coords["lon"].values  # degrees (n_lon,)

    # Time coordinate — extract UTC hour and day-of-year (vectorised)
    time_values = temperature.coords["time"].values
    dti = pd.DatetimeIndex(time_values)
    hours_utc = (dti.hour + dti.minute / 60.0).to_numpy(dtype=float)  # (n_time,)
    doy = dti.dayofyear.to_numpy(dtype=float)  # (n_time,)

    # Inverse relative distance Earth-Sun and solar declination
    # FAO-56 Eq. 23, 24
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * doy)  # (n_time,)
    dec = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)  # (n_time,) rad

    # Determine axis positions from the dims of the input array
    time_ax = temperature.dims.index("time")
    lat_ax = temperature.dims.index("lat")
    lon_ax = temperature.dims.index("lon")
    ndim = len(temperature.dims)

    def _expand(arr, axis, total_dims):
        """Insert length-1 axes so *arr* broadcasts to *total_dims* dimensions."""
        shape = [1] * total_dims
        shape[axis] = len(arr)
        return arr.reshape(shape)

    dr_b = _expand(dr, time_ax, ndim)
    dec_b = _expand(dec, time_ax, ndim)
    lat_b = _expand(lat_rad, lat_ax, ndim)
    lon_b = _expand(lon_coord, lon_ax, ndim)
    hours_b = _expand(hours_utc, time_ax, ndim)

    # Solar time angle at midpoint of the hour (FAO-56 Eq. 29 simplified for UTC):
    # ω = π/12 × (hour_utc + 0.5 + lon/15 - 12)
    omega = np.pi / 12.0 * (hours_b + 0.5 + lon_b / 15.0 - 12.0)  # rad
    omega1 = omega - np.pi / 24.0  # hour angle at start of period
    omega2 = omega + np.pi / 24.0  # hour angle at end of period

    # Ra in MJ/m²/hour (FAO-56 Eq. 28)
    gsc = 0.0820  # solar constant MJ/m²/min
    ra = (
        12.0 / np.pi
        * gsc
        * dr_b
        * (
            (omega2 - omega1) * np.sin(lat_b) * np.sin(dec_b)
            + np.cos(lat_b) * np.cos(dec_b) * (np.sin(omega2) - np.sin(omega1))
        )
    )
    ra = np.broadcast_to(ra, t.shape).copy()
    ra = np.clip(ra, 0.0, None)  # nighttime Ra = 0

    # -- Step 8: Clear-sky radiation Rso (FAO-56 Eq. 37) --
    rso = (0.75 + 2e-5 * z) * ra  # MJ/m²/hour

    # -- Step 9: Net shortwave radiation Rns (FAO-56 Eq. 38) --
    rns = (1.0 - albedo) * rs  # MJ/m²/hour

    # -- Step 10: Net longwave radiation Rnl (hourly form, FAO-56 Eq. 39) --
    # Stefan-Boltzmann constant in MJ/(m²·hour·K⁴)
    sigma_h = 2.042e-10
    t_k = t + 273.15
    humidity_term = humid_a - humid_b * np.sqrt(np.maximum(ea, 0.0))

    # Cloudiness correction — use Rs/Rso, clip to [0.25, 1.0].
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_rso_ratio = np.where(rso > 0.0, rs / rso, 0.25)
    rs_rso_ratio = np.clip(rs_rso_ratio, 0.25, 1.0)
    cloud_term = cloud_a * rs_rso_ratio - cloud_b

    rnl = sigma_h * (t_k ** 4) * humidity_term * cloud_term  # MJ/m²/hour

    # -- Step 11: Net radiation Rn (FAO-56 Eq. 40) --
    rn = rns - rnl  # MJ/m²/hour

    # -- Step 12: Daytime/nighttime determination --
    # Daytime when Rn > 0 (sun above horizon)
    is_daytime = rn > 0.0

    # -- Step 13: Soil heat flux G (FAO-56 Chapter 4) --
    g = np.where(is_daytime, g_frac_day * rn, g_frac_night * rn)

    # -- Step 14: Wind resistance coefficient Cd --
    cd = np.where(is_daytime, cd_day, cd_night)

    # -- Step 15: FAO-56 hourly Penman-Monteith equation (Eq. 53) --
    numerator = 0.408 * delta * (rn - g) + gamma * (cn / (t + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + cd * u2)

    with np.errstate(divide="ignore", invalid="ignore"):
        et0 = numerator / denominator

    et0 = np.where(np.isfinite(et0), et0, 0.0)
    et0 = np.clip(et0, 0.0, None)

    return xr.DataArray(
        et0,
        dims=temperature.dims,
        coords=temperature.coords,
        name="pet_penman_monteith_hourly_mm",
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
