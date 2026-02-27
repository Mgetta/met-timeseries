"""Cloud cover fraction derivation from downward shortwave radiation."""

from __future__ import annotations

import numpy as np
import xarray as xr


# Solar constant (W/m2)
_SOLAR_CONSTANT = 1367.0


def _solar_zenith_cos(
    time: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Compute cosine of the solar zenith angle on a latitude grid.

    Parameters
    ----------
    time : np.ndarray
        Array of :class:`numpy.datetime64` timestamps.
    lat : np.ndarray
        1-D array of latitudes in degrees.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(time), len(lat))`` containing
        ``cos(zenith)``.  Values are clipped to [0, 1].
    """
    # Day-of-year and hour (UTC) arrays
    time_dt = time.astype("datetime64[s]").astype(object)
    doy = np.array([t.timetuple().tm_yday for t in time_dt], dtype=float)
    hour_utc = np.array([t.hour + t.minute / 60.0 for t in time_dt], dtype=float)

    # Solar declination (radians)
    decl = np.deg2rad(23.45 * np.sin(np.deg2rad(360.0 / 365.0 * (doy - 81.0))))

    # Hour angle (radians) - solar noon at 12:00 UTC (rough approximation)
    ha = np.deg2rad(15.0 * (hour_utc - 12.0))

    lat_rad = np.deg2rad(lat)

    # cos(zenith) = sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(ha)
    cos_z = (
        np.sin(lat_rad)[np.newaxis, :] * np.sin(decl)[:, np.newaxis]
        + np.cos(lat_rad)[np.newaxis, :] * np.cos(decl)[:, np.newaxis] * np.cos(ha)[:, np.newaxis]
    )
    return np.clip(cos_z, 0.0, 1.0)


def compute_cloud_cover(
    dswrf: xr.DataArray,
    time: np.ndarray | None = None,
    lat: np.ndarray | None = None,
) -> xr.DataArray:
    """Compute cloud cover fraction from downward shortwave radiation.

    Estimates cloud cover as the complement of the clearness index::

        C = 1 - SW_actual / SW_clear

    where the clear-sky radiation is estimated from the solar constant and
    solar geometry.  Cloud cover is set to ``NaN`` during night-time hours
    (when the sun is below the horizon).

    Parameters
    ----------
    dswrf : xr.DataArray
        Downward shortwave radiation flux in W/m2 with dimensions
        ``(time, lat, lon)`` (or a subset thereof).
    time : np.ndarray or None
        Array of :class:`numpy.datetime64` timestamps.  If ``None``,
        extracted from ``dswrf.time``.
    lat : np.ndarray or None
        1-D array of latitudes in degrees.  If ``None``, extracted from
        ``dswrf.lat`` or ``dswrf.y``.

    Returns
    -------
    xr.DataArray
        Cloud cover fraction in the range [0, 1] during daytime.
        Night-time values are ``NaN``.  Carries ``name="cloud_cover"``
        and ``attrs["units"] = "1"`` (dimensionless fraction).
    """
    if time is None:
        time = dswrf.time.values
    if lat is None:
        lat_coord = "lat" if "lat" in dswrf.coords else "y"
        lat = dswrf[lat_coord].values

    cos_z = _solar_zenith_cos(time, lat)  # (time, lat)

    # Clear-sky radiation = solar_constant * cos(zenith)
    # Expand to (time, lat, lon) if needed
    if dswrf.ndim == 3:
        cos_z = cos_z[:, :, np.newaxis]  # broadcast over lon

    sw_clear = _SOLAR_CONSTANT * cos_z

    # Clearness index
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = np.where(sw_clear > 0, dswrf.values / sw_clear, np.nan)

    # Cloud cover = 1 - kt, clipped to [0, 1]
    cloud_cover_vals = np.clip(1.0 - kt, 0.0, 1.0)

    # Night-time: sun below horizon -> NaN
    night_mask = sw_clear <= 0
    cloud_cover_vals = np.where(night_mask, np.nan, cloud_cover_vals)

    result = xr.DataArray(
        cloud_cover_vals,
        coords=dswrf.coords,
        dims=dswrf.dims,
        name="cloud_cover",
        attrs={"units": "1", "long_name": "Cloud cover fraction"},
    )
    return result
