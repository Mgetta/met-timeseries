"""
NLDAS-2 data source.

The public API is :func:`fetch_nldas`, which downloads NLDAS-2 hourly forcings
for a given bounding box, year, and month and returns an aggregated monthly
:class:`xarray.Dataset`.

Data access uses the ``earthaccess`` library, which handles NASA Earthdata
authentication automatically via ``~/.netrc``, environment variables
(``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``), or an interactive prompt.
Users need a free NASA Earthdata account: https://urs.earthdata.nasa.gov
"""

from __future__ import annotations

import calendar
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from met_timeseries.sources.base import BoundingBox

logger = logging.getLogger(__name__)

#: NLDAS-2 variables available via the forcing-A product
AVAILABLE_VARIABLES: list[str] = [
    "TMP",       # 2-m air temperature (K)
    "SPFH",      # 2-m specific humidity (kg/kg)
    "PRES",      # surface pressure (Pa)
    "UGRD",      # 10-m zonal wind (m/s)
    "VGRD",      # 10-m meridional wind (m/s)
    "DLWRF",     # downward longwave radiation (W/m²)
    "CONVfrac",  # convective fraction (-)
    "CAPE",      # convective available potential energy (J/kg)
    "PEVAP",     # potential evaporation (kg/m²)
    "APCP",      # precipitation hourly total (kg/m²)
    "DSWRF",     # downward shortwave radiation (W/m²)
]


def fetch_nldas(
    bounds: BoundingBox,
    year: int,
    month: int,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
) -> xr.Dataset:
    """Fetch NLDAS-2 hourly forcing data for the given bounding box.

    Authentication is handled automatically by :func:`earthaccess.login`,
    which reads ``~/.netrc``, environment variables (``EARTHDATA_USERNAME`` /
    ``EARTHDATA_PASSWORD``), or prompts interactively.  A free NASA Earthdata
    account is required: https://urs.earthdata.nasa.gov

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    year:
        Calendar year (e.g. 2010).
    month:
        Calendar month (1–12).
    variables:
        NLDAS-2 short variable names to include.  Defaults to ``["APCP",
        "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    cache_dir:
        If provided, a subsetted monthly NetCDF file is cached here and
        loaded on subsequent calls for the same month.

    Returns
    -------
    xarray.Dataset
        Hourly Dataset with ``time``, ``lat``, and ``lon`` dimensions and the
        requested variables.  The ``time`` coordinate carries proper
        ``datetime64`` timestamps.

    Raises
    ------
    RuntimeError
        If no granules are found for the requested period.
    """
    import earthaccess

    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    logger.info("Fetching NLDAS-2 data: year=%d month=%d bounds=%r", year, month, bounds)

    # Check cache first
    if cache_dir is not None:
        cache_path = _monthly_cache_path(cache_dir, year, month)
        if cache_path.exists():
            logger.debug("Loading NLDAS-2 from cache: %s", cache_path)
            return xr.open_dataset(cache_path)

    # Authenticate with NASA Earthdata
    earthaccess.login()

    # Temporal range: full month
    _, n_days = calendar.monthrange(year, month)
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{n_days:02d}"

    # Search for NLDAS-2 granules
    results = earthaccess.search_data(
        short_name="NLDAS_FORA0125_H",
        version="2.0",
        temporal=(start, end),
        bounding_box=(bounds.west, bounds.south, bounds.east, bounds.north),
    )

    if not results:
        raise RuntimeError(f"No NLDAS-2 granules found for {year}-{month:02d}")

    logger.debug("Found %d NLDAS-2 granules", len(results))

    # Open all granules as fsspec file objects
    file_objs = earthaccess.open(results)

    # Load all granules into a single Dataset
    ds = xr.open_mfdataset(
        file_objs,
        engine="h5netcdf",
        combine="nested",
        concat_dim="time",
    )

    # Subset variables
    ds = ds[variables]

    # Subset spatial domain
    ds = ds.sel(
        lat=slice(bounds.south, bounds.north),
        lon=slice(bounds.west, bounds.east),
    )

    # Ensure time coordinate has proper datetime64 values
    if "time" not in ds.coords or not np.issubdtype(ds["time"].dtype, np.datetime64):
        n_hours = ds.sizes["time"]
        start_ts = np.datetime64(dt.datetime(year, month, 1), "ns")
        timestamps = [start_ts + np.timedelta64(h, "h") for h in range(n_hours)]
        ds = ds.assign_coords(time=timestamps)

    # Save to cache
    if cache_dir is not None:
        cache_path = _monthly_cache_path(cache_dir, year, month)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache_path)
        logger.debug("Cached NLDAS-2 data to: %s", cache_path)

    return ds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _monthly_cache_path(cache_dir: str, year: int, month: int) -> Path:
    return Path(cache_dir) / "nldas" / f"{year}{month:02d}.nc"
