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
        ``DSWRF`` (shortwave radiation).

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
