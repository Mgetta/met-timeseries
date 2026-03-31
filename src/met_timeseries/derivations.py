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
        dp.to("degF").magnitude,
        dims=specific_humidity.dims,
        coords=specific_humidity.coords,
        name="dewpoint_f",
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
