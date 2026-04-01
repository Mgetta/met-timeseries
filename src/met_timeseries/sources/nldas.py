"""
NLDAS-2 data source.

Architecture
------------
The module separates data access into three layers:

**Download layer** – acquires raw per-cell timeseries:
    :func:`download_datarods` fetches raw hourly data from the Giovanni
    Time Series API for every NLDAS grid cell within a bounding box.
    It is independent of any weighting and caches raw per-cell CSVs.

**Processing layer** – spatial aggregation:
    :func:`process_nldas` is the main entry point for the datarods
    workflow.  It calls :func:`download_datarods`, derives area-overlap
    weights automatically from the bounding box and NLDAS grid (via
    :func:`compute_nldas_weights`), and computes polygon-level
    weighted-average timeseries via :func:`compute_weighted_averages`.
    Pre-computed weights may also be supplied.

**Grid download layer** – full gridded NetCDF access:
    :func:`fetch_nldas_grid` downloads full NLDAS-2 gridded files via
    ``earthaccess`` and returns an :class:`xarray.Dataset`.

Data access uses the ``earthaccess`` library, which handles NASA Earthdata
authentication automatically via ``~/.netrc``, environment variables
(``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``), or an interactive prompt.
Users need a free NASA Earthdata account: https://urs.earthdata.nasa.gov
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import io
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from met_timeseries.sources.base import CACHE_BOUNDS, BoundingBox

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

_CACHE_PREFIX = "NLDAS_FORA0125_H_"

#: NLDAS-2 grid resolution in degrees; used as the cell buffer when selecting
#: grid cells that fall within a bounding box.
_NLDAS_RESOLUTION: float = 0.125


def fetch_nldas(
    date: str,
    cache_dir: str,
    bounds: BoundingBox | None = None,
    variables: list[str] = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"],
    max_connections: int = 8,
) -> xr.Dataset:
    """Fetch NLDAS-2 hourly gridded data for the given bounding box.

    Downloads full NLDAS-2 gridded NetCDF files via ``earthaccess`` and
    returns a subsetted :class:`xarray.Dataset`.

    Authentication is handled automatically by :func:`earthaccess.login`,
    which reads ``~/.netrc``, environment variables (``EARTHDATA_USERNAME`` /
    ``EARTHDATA_PASSWORD``), or prompts interactively.  A free NASA Earthdata
    account is required: https://urs.earthdata.nasa.gov

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    start:
        Start date in ``"YYYY-MM-DD"`` format.
    end:
        End date in ``"YYYY-MM-DD"`` format (inclusive).  When ``None``
        (the default), only the single day given by *start* is fetched.
    variables:
        NLDAS-2 short variable names to include.  Defaults to ``["APCP",
        "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    max_connections:
        Maximum number of concurrent granule downloads.  Defaults to ``8``.

    Returns
    -------
    xarray.Dataset
        Hourly Dataset with ``time``, ``lat``, and ``lon`` dimensions.
        The ``time`` coordinate carries proper ``datetime64`` timestamps.

    Raises
    ------
    RuntimeError
        If no granules are found for a requested period.
    """
 

    if bounds is None:
        bounds = CACHE_BOUNDS

    date = dt.date.fromisoformat(date)
    
    cache_path = Path(cache_dir) / f"{_CACHE_PREFIX}{date.strftime('%Y%m%d')}.nc"

    if cache_path.exists():
        ds = xr.open_dataset(cache_path)
        missing_vars = [var for var in variables if var not in ds.data_vars]
        if missing_vars: # Add missing variables to the existing dataset
            ds_missing = download(str(date.date()), 
                                  str(date.date()), 
                                  missing_vars, 
                                  max_connections)
            ds = xr.merge([ds, ds_missing])
            _cache_dataset(ds, cache_path)
    else:
        ds = download(str(date.date()), 
                      str(date.date()), 
                      variables, 
                      max_connections
        )
        _cache_dataset(ds, cache_path)

    ds = _clip_dataset(ds, bounds=bounds)
    return ds


def download(
    start_date: dt.date,
    end_date: dt.date,
    variables: list[str],
    max_connections: int) -> xr.Dataset:
    """Download and subset all granules for a date range, returning a Dataset.

    Opens granules concurrently using :class:`concurrent.futures.ThreadPoolExecutor`.
    Failed granules are skipped with a warning; a :exc:`RuntimeError` is raised
    only when *all* granules for the range fail.

    Parameters
    ----------
    bounds: BoundingBox | None = None
        Spatial bounding box in EPSG:4326. Defaults to MN_BOUNDS if None.
    start_date:
        Start of the temporal range in ``"YYYY-MM-DD"`` format.
    end_date:
        End of the temporal range in ``"YYYY-MM-DD"`` format (inclusive).
    variables:
        Variable names to include.
    max_connections:
        Maximum concurrent granule downloads.
    cache_dir:
        Directory to cache downloaded granules. If None, caching is disabled.
    max_connections:
        Maximum concurrent granule downloads.

    Returns
    -------
    xarray.Dataset
        Dataset concatenated along the ``time`` dimension.
    """
    import earthaccess
    earthaccess.login()
    results = _search_nldas_granules(start_date, end_date)
    file_objs = earthaccess.open(results)

    granule_datasets: list[xr.Dataset] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_connections) as executor:
        futures = {
            executor.submit(_open_file_obj, fobj): i
            for i, fobj in enumerate(file_objs)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                granule_datasets.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping granule %d for %s to %s due to error: %s",
                    idx, start_date, end_date, exc,
                )

    if not granule_datasets:
        raise RuntimeError(
            f"All granules failed for {start_date} to {end_date}"
        )

    ds = xr.concat(
        sorted(granule_datasets, key=lambda d: d["time"].values[0]),
        dim="time",
    )

    ds = _clip_dataset(ds,CACHE_BOUNDS)
    # Some NLDAS-2 granules store time as a numeric offset rather than
    # datetime64. Reconstruct the coordinate from the known range start if
    # the decoded dtype is not already datetime64.
    if "time" not in ds.coords or not np.issubdtype(ds["time"].dtype, np.datetime64):
        n_hours = ds.sizes["time"]
        start_dt = start_date
        start_ts = np.datetime64(start_dt, "ns")
        timestamps = [start_ts + np.timedelta64(h, "h") for h in range(n_hours)]
        ds = ds.assign_coords(time=timestamps)

    return ds

def get_nldas_gridcells(bounds: BoundingBox) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of NLDAS grid cells that intersect the bounding box."""
    grid = _generate_nldas_grid(
        west=bounds.west - _NLDAS_RESOLUTION,
        south=bounds.south - _NLDAS_RESOLUTION,
        east=bounds.east + _NLDAS_RESOLUTION,
        north=bounds.north + _NLDAS_RESOLUTION,
    )
    bbox_polygon = _bounds_to_polygon(bounds)
    grid_in_bounds = grid[grid.intersects(bbox_polygon)]
    return grid_in_bounds[["lat_center", "lon_center", "geometry"]].drop_duplicates()


def _cache_dataset(ds: xr.Dataset,cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": 9,"shuffle": True}
        for var in ds.data_vars
    }
    ds.to_netcdf(cache_path, encoding=encoding)
    logger.debug("Cached NLDAS data to %s", cache_path)
    return cache_path


def _bounds_to_polygon(bounds: BoundingBox):
    """Return a Shapely box polygon for *bounds*."""
    return box(bounds.west, bounds.south, bounds.east, bounds.north)


def _open_file_obj(file_obj: object) -> xr.Dataset:
    """Open a file-like object as an xarray Dataset. Note that this is an in-memory operation and does not write to disk."""
    return xr.open_dataset(file_obj, engine="h5netcdf")


def _clip_dataset(
    ds: xr.Dataset,
    bounds: BoundingBox
) -> xr.Dataset:
    """
    Clip an xarray Dataset to the given spatial bounding box.

    Parameters
    ----------
    ds:
        An xarray Dataset representing an NLDAS-2 hourly NetCDF granule.
    bounds:
        Spatial bounding box; only data within these bounds is returned.

    Returns
    -------
    xarray.Dataset
        In-memory Dataset containing only the spatial extent
        defined by *bounds*.
    """

    ds = ds.sel(
        lat=slice(bounds.south, bounds.north),
        lon=slice(bounds.west, bounds.east),
    )
    ds = ds.load()
    return ds


def _search_nldas_granules(
    start_date: dt.date,
    end_date: dt.date,
    bounds: BoundingBox | None = None,
) -> list:
    """Search for NLDAS-2 granules for a given date range via CMR.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    start_date:
        Start of the temporal range in ``"YYYY-MM-DD"`` format.
    end_date:
        End of the temporal range in ``"YYYY-MM-DD"`` format (inclusive).

    Returns
    -------
    list
        Granule results from ``earthaccess.search_data``.

    Raises
    ------
    RuntimeError
        If no granules are found.
    """
    import earthaccess


    if bounds is None:
        bounds = CACHE_BOUNDS

    results = earthaccess.search_data(
        short_name="NLDAS_FORA0125_H",
        version="2.0",
        temporal=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        bounding_box=(bounds.west, bounds.south, bounds.east, bounds.north),
    )

    if not results:
        raise RuntimeError(
            f"No NLDAS-2 granules found for {start_date} to {end_date}"
        )

    logger.debug(
        "Found %d NLDAS-2 granules for %s to %s", len(results), start_date, end_date
    )
    return results


# ---------------------------------------------------------------------------
# Grid generation and weight computation
# ---------------------------------------------------------------------------

def _generate_nldas_grid(
    west: float = -124.9375,
    south: float = 25.0625,
    east: float = -67.0625,
    north: float = 52.9375,
    resolution: float = 0.125,
) -> gpd.GeoDataFrame:
    """Generate the NLDAS 0.125° grid as a GeoDataFrame.

    Each row represents one grid cell as a polygon, with ``lat_center``
    and ``lon_center`` columns giving the cell centroid coordinates.

    The default extent covers the CONUS NLDAS-2 domain.

    Parameters
    ----------
    west, south, east, north:
        Grid extent in EPSG:4326 degrees. Defaults to the full NLDAS-2 domain.
    resolution:
        Grid cell size in degrees. Default 0.125°.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with columns ``lat_center``, ``lon_center``, ``geometry``
        in EPSG:4326.
    """
    half = resolution / 2.0
    lons = np.arange(west, east + half, resolution)
    lats = np.arange(south, north + half, resolution)

    records = []
    for lon in lons:
        for lat in lats:
            cell = box(lon - half, lat - half, lon + half, lat + half)
            records.append(
                {
                    "lon_center": round(lon, 4),
                    "lat_center": round(lat, 4),
                    "geometry": cell,
                }
            )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


