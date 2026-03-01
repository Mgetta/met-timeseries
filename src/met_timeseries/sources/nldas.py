"""
NLDAS-2 data source.

The public API is :func:`fetch_nldas`, which downloads NLDAS-2 hourly forcings
for a given bounding box, year, and month and returns an aggregated monthly
:class:`xarray.Dataset`.

Data access uses the NASA GES DISC OPeNDAP endpoint.  Users must have a
valid NASA Earthdata account and a ``.netrc`` / ``~/.dodsrc`` configuration
for authenticated access.  See:
https://disc.gsfc.nasa.gov/data-access#python-requests
"""

from __future__ import annotations

import calendar
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from met_timeseries.sources.base import BoundingBox

logger = logging.getLogger(__name__)

_OPENDAP_TEMPLATE = (
    "https://hydro1.gesdisc.eosdis.nasa.gov/opendap/NLDAS/NLDAS_FORA0125_H.2.0/"
    "{year}/{doy:03d}/NLDAS_FORA0125_H.A{year}{month:02d}{day:02d}.{hour:04d}.002.grb.SUB.nc"
)

#: NLDAS-2 variables available via the forcing-A product
AVAILABLE_VARIABLES: list[str] = [
    "TMP",   # 2-m air temperature (K)
    "SPFH",  # 2-m specific humidity (kg/kg)
    "PRES",  # surface pressure (Pa)
    "UGRD",  # 10-m zonal wind (m/s)
    "VGRD",  # 10-m meridional wind (m/s)
    "DLWRF", # downward longwave radiation (W/m²)
    "CONVfrac",  # convective fraction (-)
    "CAPE",  # convective available potential energy (J/kg)
    "PEVAP", # potential evaporation (kg/m²)
    "APCP",  # precipitation hourly total (kg/m²)
    "DSWRF", # downward shortwave radiation (W/m²)
]


def fetch_nldas(
    bounds: BoundingBox,
    year: int,
    month: int,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
) -> xr.Dataset:
    """Fetch NLDAS-2 monthly forcing data for the given bounding box.

    Downloads hourly data for each day of *month*/*year*, subsets to
    *bounds*, and returns a Dataset with one variable per requested forcing
    containing the monthly mean (or total for precipitation).

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
        "TMP", "DSWRF"]``.
    cache_dir:
        If provided, downloaded files are cached here to avoid repeated
        downloads.

    Returns
    -------
    xarray.Dataset
        Monthly Dataset with ``lat`` and ``lon`` dimensions and the requested
        variables.

    Raises
    ------
    RuntimeError
        If data cannot be fetched for the requested period.
    """
    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP"]

    logger.info("Fetching NLDAS-2 data: year=%d month=%d bounds=%r", year, month, bounds)

    datasets: list[xr.Dataset] = []
    _, n_days = calendar.monthrange(year, month)

    for day in range(1, n_days + 1):
        day_of_year = _day_of_year(year, month, day)
        ds_day = _fetch_nldas_day(bounds, year, month, day, day_of_year, variables, cache_dir)
        datasets.append(ds_day)

    if not datasets:
        raise RuntimeError(f"No NLDAS data retrieved for {year}-{month:02d}")

    combined = xr.concat(datasets, dim="time")

    # Monthly aggregation: sum precipitation, mean everything else
    result_vars: dict[str, xr.DataArray] = {}
    for var in variables:
        if var in ("APCP", "PEVAP"):
            result_vars[var] = combined[var].sum(dim="time")
        else:
            result_vars[var] = combined[var].mean(dim="time")

    return xr.Dataset(result_vars)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _day_of_year(year: int, month: int, day: int) -> int:
    import datetime

    return datetime.date(year, month, day).timetuple().tm_yday


def _fetch_nldas_day(
    bounds: BoundingBox,
    year: int,
    month: int,
    day: int,
    day_of_year: int,
    variables: list[str],
    cache_dir: str | None,
) -> xr.Dataset:
    """Fetch and aggregate a single day of NLDAS-2 data."""
    hourly: list[xr.Dataset] = []
    for hour in range(0, 2400, 100):
        url = _OPENDAP_TEMPLATE.format(
            year=year, doy=day_of_year, month=month, day=day, hour=hour
        )
        cache_path = _cache_path(cache_dir, url) if cache_dir else None
        ds_hour = _open_dataset(url, bounds, variables, cache_path)
        hourly.append(ds_hour)

    return xr.concat(hourly, dim="time")


def _cache_path(cache_dir: str, url: str) -> Path:
    import hashlib

    digest = hashlib.md5(url.encode()).hexdigest()  # noqa: S324
    return Path(cache_dir) / "nldas" / f"{digest}.nc"


def _open_dataset(
    url: str,
    bounds: BoundingBox,
    variables: list[str],
    cache_path: Path | None,
) -> xr.Dataset:
    """Open (and optionally cache) a single NLDAS granule."""
    if cache_path is not None and cache_path.exists():
        return xr.open_dataset(cache_path)

    ds = xr.open_dataset(url, engine="netcdf4")
    ds = ds[variables]
    ds = ds.sel(
        lat=slice(bounds.south, bounds.north),
        lon=slice(bounds.west, bounds.east),
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache_path)

    return ds
