"""
Derived variable calculations for NLDAS-2 forcings.

Each function accepts raw NLDAS-2 :class:`xarray.Dataset` variables and
returns a derived :class:`xarray.DataArray`.
"""

from __future__ import annotations

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

    if "TMP" in nldas_data and "SPFH" in nldas_data:
        derived["dewpoint_c"] = _dewpoint(nldas_data["TMP"], nldas_data["SPFH"])

    if "PEVAP" in nldas_data:
        # PEVAP is potential evaporation in kg/m², numerically equal to mm
        derived["pet_mm"] = nldas_data["PEVAP"].rename("pet_mm")
    elif "TMP" in nldas_data and "DSWRF" in nldas_data:
        derived["pet_hargreaves_mm"] = _hargreaves_pet(
            nldas_data["TMP"], nldas_data["DSWRF"]
        )

    return derived


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
