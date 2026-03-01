"""
Derived variable calculations for NLDAS-2 forcings.

Each function accepts raw NLDAS-2 :class:`xarray.Dataset` variables and
returns a derived :class:`xarray.DataArray`.
"""

from __future__ import annotations

import datetime

import numpy as np
import xarray as xr


def derive_variables(nldas_data: xr.Dataset) -> dict[str, xr.DataArray]:
    """Derive secondary variables from raw NLDAS-2 output.

    Parameters
    ----------
    nldas_data:
        Dataset produced by :func:`~met_timeseries.sources.nldas.fetch_nldas`
        containing at minimum ``APCP`` (precip), ``TMP`` (temperature), and
        ``DSWRF`` (shortwave radiation).  If ``PEVAP`` is present it is used
        directly as ``pet_mm``; otherwise a Hargreaves-based estimate is
        computed when ``TMP`` and ``DSWRF`` are available.

    Returns
    -------
    dict mapping derived variable name to :class:`xarray.DataArray`.
    """
    derived: dict[str, xr.DataArray] = {}

    if "APCP" in nldas_data:
        derived["precip_mm"] = nldas_data["APCP"].rename("precip_mm")

    if "TMP" in nldas_data:
        # Convert Kelvin to Celsius
        derived["temp_c"] = (nldas_data["TMP"] - 273.15).rename("temp_c")

    if "DSWRF" in nldas_data:
        derived["shortwave_wm2"] = nldas_data["DSWRF"].rename("shortwave_wm2")
        derived["cloud_cover_fraction"] = _cloud_cover(nldas_data)

    if "TMP" in nldas_data and "SPFH" in nldas_data:
        derived["dewpoint_c"] = _dewpoint(nldas_data["TMP"], nldas_data["SPFH"])

    if "PEVAP" in nldas_data:
        # PEVAP is potential evaporation in kg/m², numerically equal to mm
        derived["pet_mm"] = nldas_data["PEVAP"].rename("pet_mm")
    elif "TMP" in nldas_data and "DSWRF" in nldas_data:
        derived["pet_hargreaves_mm"] = _hargreaves_pet(
            nldas_data["TMP"], nldas_data["DSWRF"]
        )
    if "UGRD" in nldas_data and "VGRD" in nldas_data:
        derived["wind_speed_ms"] = (
            np.sqrt(nldas_data["UGRD"] ** 2 + nldas_data["VGRD"] ** 2)
        ).rename("wind_speed_ms")

    return derived


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clear_sky_radiation(lat: float, lon: float, dt: datetime.datetime) -> float:
    """Compute theoretical clear-sky surface shortwave radiation (W/m²).

    Uses standard solar geometry with a fixed atmospheric transmittance.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees.
    lon:
        Longitude in decimal degrees (unused beyond type-checking; retained for
        future equation-of-time corrections).
    dt:
        UTC datetime for which to compute the radiation.

    Returns
    -------
    Clear-sky surface shortwave radiation in W/m², or 0.0 at nighttime.
    """
    S0 = 1361.0  # solar constant (W/m²)
    tau = 0.75   # clear-sky atmospheric transmittance

    n = dt.timetuple().tm_yday  # day of year

    # Solar declination (degrees)
    dec_deg = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + n)))
    dec_rad = np.radians(dec_deg)

    # Hour angle: 15° per hour from solar noon (UTC hour used as approximation)
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


def _cloud_cover(nldas_data: xr.Dataset) -> xr.DataArray:
    """Estimate cloud-cover fraction from DSWRF and theoretical clear-sky radiation.

    The fraction is defined as ``1 - DSWRF / R_clearsky`` and is clamped to
    ``[0, 1]``.  Nighttime pixels (clear-sky < 10 W/m²) are set to NaN.

    Parameters
    ----------
    nldas_data:
        Dataset containing at least ``DSWRF`` and spatial coordinates
        ``lat``/``lon``.  A ``time`` dimension is used when present to compute
        solar geometry per time step; otherwise the current UTC time is used.

    Returns
    -------
    :class:`xarray.DataArray` named ``cloud_cover_fraction``.
    """
    dswrf: xr.DataArray = nldas_data["DSWRF"]

    # Determine the timestamp to use for solar-geometry calculations.
    if "time" in nldas_data.dims or "time" in nldas_data.coords:
        # Build a clear-sky array that may vary over time.
        time_values = nldas_data["time"].values
        # Work time step by time step; result is the same shape as dswrf.
        clearsky_values = np.full(dswrf.shape, np.nan, dtype=float)

        # Identify axis positions so we can iterate over time regardless of
        # dimension order.
        time_axis = dswrf.dims.index("time")
        lat_axis = dswrf.dims.index("lat") if "lat" in dswrf.dims else None
        lon_axis = dswrf.dims.index("lon") if "lon" in dswrf.dims else None

        lats = nldas_data["lat"].values if "lat" in nldas_data.coords else np.array([0.0])
        lons = nldas_data["lon"].values if "lon" in nldas_data.coords else np.array([0.0])

        for t_idx, t_val in enumerate(time_values):
            dt = _to_datetime(t_val)
            # Build a 2-D (lat × lon) slice of clear-sky values for this time.
            cs_slice = np.array(
                [[_clear_sky_radiation(la, lo, dt) for lo in lons] for la in lats],
                dtype=float,
            )
            # Insert into the full array at the correct time index.
            idx: list = [slice(None)] * clearsky_values.ndim
            idx[time_axis] = t_idx
            if lat_axis is not None and lon_axis is not None:
                # cs_slice shape is (n_lat, n_lon); reorder axes to match dswrf.
                # After fixing the time axis to t_idx, the remaining axes shift
                # down by 1 for any axis that was originally after time_axis.
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

        clearsky = xr.DataArray(clearsky_values, dims=dswrf.dims, coords=dswrf.coords)
    else:
        # No time dimension — use a fixed reference (solar noon, mid-year) so
        # that results are reproducible regardless of when the code is run.
        dt = datetime.datetime(2000, 7, 1, 12, 0, 0)
        lats = nldas_data["lat"].values if "lat" in nldas_data.coords else np.array([0.0])
        lons = nldas_data["lon"].values if "lon" in nldas_data.coords else np.array([0.0])
        cs_values = np.array(
            [[_clear_sky_radiation(la, lo, dt) for lo in lons] for la in lats],
            dtype=float,
        )
        lat_dim = "lat" if "lat" in dswrf.dims else dswrf.dims[0]
        lon_dim = "lon" if "lon" in dswrf.dims else dswrf.dims[1]
        clearsky = xr.DataArray(
            cs_values,
            dims=[lat_dim, lon_dim],
            coords={lat_dim: nldas_data.coords.get(lat_dim), lon_dim: nldas_data.coords.get(lon_dim)},
        )

    # Mask pixels where clear-sky radiation is too low (nighttime / near-horizon).
    _MIN_CLEARSKY = 10.0
    daytime_mask = clearsky > _MIN_CLEARSKY

    with np.errstate(divide="ignore", invalid="ignore"):
        cloud_cover = 1.0 - dswrf / clearsky

    # Clamp to [0, 1] and apply nighttime mask.
    cloud_cover = cloud_cover.clip(0.0, 1.0)
    cloud_cover = cloud_cover.where(daytime_mask)

    return cloud_cover.rename("cloud_cover_fraction")


def _to_datetime(t_val) -> datetime.datetime:
    """Convert an arbitrary numpy/pandas timestamp to a :class:`datetime.datetime`."""
    # numpy datetime64 → pandas Timestamp → stdlib datetime
    import pandas as pd

    return pd.Timestamp(t_val).to_pydatetime(warn=False)


def _dewpoint(temp_k: xr.DataArray, spfh: xr.DataArray) -> xr.DataArray:
    """Estimate dew-point temperature (°C) from temperature and specific humidity."""
    # Approximate vapour pressure from specific humidity and standard pressure
    pres_pa = 101325.0
    e = spfh * pres_pa / (0.622 + spfh)
    # Magnus formula approximation
    with np.errstate(divide="ignore", invalid="ignore"):
        dp_c = (243.04 * np.log(e / 611.2)) / (17.625 - np.log(e / 611.2))
    return dp_c.rename("dewpoint_c")


def _hargreaves_pet(temp_k: xr.DataArray, dswrf: xr.DataArray) -> xr.DataArray:
    """Estimate PET (mm) using a simplified Hargreaves (1985) method.

    This is a fallback for when NLDAS-2 ``PEVAP`` is not available.  The
    standard Hargreaves equation requires daily min/max temperatures to
    estimate the diurnal temperature range.  Because NLDAS-2 provides only a
    single temperature field (``TMP`), the diurnal range is approximated by a
    climatological default of 10 °C, which is representative of mid-latitude
    continental conditions.

    Formula (simplified):
        PET ≈ 0.0023 × DSWRF × (T_celsius + 17.8) × sqrt(diurnal_range)

    where DSWRF is used as a proxy for extraterrestrial radiation (Ra) scaled
    to the same units, and diurnal_range defaults to 10 °C.

    Assumptions
    -----------
    * Diurnal temperature range is approximated as 10 °C.
    * DSWRF (W/m²) is used as a radiation proxy in place of Ra.
    * Negative PET values are clipped to 0.

    Parameters
    ----------
    temp_k:
        Air temperature in Kelvin (``TMP`` from NLDAS-2).
    dswrf:
        Downward shortwave radiation in W/m² (``DSWRF`` from NLDAS-2).

    Returns
    -------
    xarray.DataArray
        Estimated PET in mm, named ``pet_hargreaves_mm``.
    """
    _DIURNAL_RANGE_DEFAULT_C = 10.0  # °C
    temp_c = temp_k - 273.15
    pet = 0.0023 * dswrf * (temp_c + 17.8) * np.sqrt(_DIURNAL_RANGE_DEFAULT_C)
    pet = pet.clip(min=0.0)
    return pet.rename("pet_hargreaves_mm")
