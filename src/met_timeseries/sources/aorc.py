"""
AORC (Analysis of Record for Calibration) data source.

Architecture
------------
This module provides data access to the NOAA AORC v1.1 dataset hosted on AWS Open Data.
Unlike NLDAS which uses earthaccess to download NetCDF granules, AORC is stored as 
cloud-optimized Zarr stores (chunked by year) natively on Amazon S3.

The spatial and temporal subsetting is performed lazily using xarray and fsspec, 
meaning only the chunks intersecting your bounding box and time period are downloaded 
into memory.

Dependencies:
    Requires `fsspec`, `s3fs`, and `zarr` to map the remote AWS S3 stores.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

from met_timeseries.sources.base import CACHE_BOUNDS, BoundingBox

logger = logging.getLogger(__name__)

# Native AORC v1.1 variable names available via AWS Open Data
AVAILABLE_VARIABLES: list[str] = [
    "APCP_surface",         # Hourly total precipitation (kg/m^2)
    "TMP_2maboveground",    # Air Temperature 2m AGL (K)
    "SPFH_2maboveground",   # Specific Humidity 2m AGL (kg/kg)
    "PRES_surface",         # Surface Pressure (Pa)
    "UGRD_10maboveground",  # U-Wind 10m AGL (m/s)
    "VGRD_10maboveground",  # V-Wind 10m AGL (m/s)
    "DLWRF_surface",        # Downward Longwave Radiation (W/m^2)
    "DSWRF_surface",        # Downward Shortwave Radiation (W/m^2)
]

_CACHE_PREFIX = "AORC_1KM_H_"
_AORC_S3_BUCKET = "s3://noaa-nws-aorc-v1-1-1km"

# AORC grid resolution in degrees (30 arc-seconds)
_AORC_RESOLUTION: float = 1.0 / 120.0


def fetch_aorc(
    date: str,
    cache_dir: str | Path,
    bounds: BoundingBox | None = None,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Fetch daily AORC hourly gridded data for the given bounding box.

    Downloads AORC Zarr chunks from AWS via fsspec and returns a subsetted
    :class:`xarray.Dataset`. Caches the subsetted day locally as a NetCDF file.

    Parameters
    ----------
    date:
        Target date in ``"YYYY-MM-DD"`` format.
    cache_dir:
        Directory in which to persist downloaded NetCDF caches.
    bounds:
        Spatial bounding box in EPSG:4326.
    variables:
        AORC variable short names to include. Defaults to ``None``, which
        fetches all available energy-balance variables.

    Returns
    -------
    xarray.Dataset
        Hourly Dataset with ``time``, ``lat``, and ``lon`` dimensions.
    """
    if variables is None:
        variables = AVAILABLE_VARIABLES

    target_date = dt.date.fromisoformat(date)
    cache_path = Path(cache_dir) / f"{_CACHE_PREFIX}{target_date.strftime('%Y%m%d')}.nc"

    if cache_path.exists():
        ds = _read_from_cache(cache_path)
        missing_vars = [var for var in variables if var not in ds.data_vars]
        if missing_vars:
            logger.info("Fetching missing variables %s for AORC date %s", missing_vars, date)
            ds_missing = download(target_date, missing_vars, bounds=bounds)
            ds_cached = ds.load()
            ds.close()
            ds = xr.merge([ds_cached, ds_missing])
            _write_to_cache(ds, cache_path)
    else:
        ds = download(target_date, variables, bounds=bounds)
        _write_to_cache(ds, cache_path)

    # In case the cached version covers a larger bounding box than currently requested
    if bounds is not None:
        ds = _clip_dataset(ds, bounds=bounds)

    return ds.load()


def download(
    target_date: dt.date,
    variables: list[str],
    bounds: BoundingBox | None = None,
) -> xr.Dataset:
    """Download AORC data lazily from AWS Zarr and subset to bounds/date.

    Opens the yearly Zarr store on AWS via fsspec, lazily selects the date and 
    spatial bounds, and triggers the chunk downloads into memory.
    """
    if bounds is None:
        bounds = CACHE_BOUNDS

    # AORC data is partitioned into yearly Zarr stores
    year = target_date.year
    s3_url = f"{_AORC_S3_BUCKET}/{year}.zarr/"
    
    logger.debug("Mapping AORC Zarr store from %s", s3_url)
    
    # Use fsspec to map the S3 bucket without credentials (anon=True)
    mapper = fsspec.get_mapper(s3_url, anon=True)
    
    # Open the Zarr store lazily. AORC 1.1 uses consolidated metadata.
    try:
        ds = xr.open_zarr(mapper, consolidated=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to open AORC Zarr store for {year}: {exc}") from exc

    # Standardize coordinate names to match the NLDAS/PRISM pipeline convention
    rename_map = {}
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    # Subset the dataset lazily
    # 1. Variables
    valid_vars = [v for v in variables if v in ds.data_vars]
    if not valid_vars:
        raise ValueError(f"None of the requested variables {variables} exist in the AORC store.")
    ds = ds[valid_vars]

    # 2. Spatial subset (done prior to time subsetting to optimize chunk lookups)
    ds = _clip_dataset(ds, bounds)

    # 3. Temporal subset 
    # Attempting string-based native xarray slicing first
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        ds = ds.sel(time=date_str)
    except KeyError:
        # Fallback to precise bounding if index structure rejects string slicing
        start_time = np.datetime64(target_date, 'ns')
        end_time = start_time + np.timedelta64(1, 'D') - np.timedelta64(1, 'ns')
        ds = ds.sel(time=slice(start_time, end_time))

    # Trigger the actual Dask chunk downloads into memory over the subset footprint
    logger.info("Downloading AORC chunks for %s over bounding box...", target_date)
    ds = ds.load()
    
    return ds


def _clip_dataset(
    ds: xr.Dataset,
    bounds: BoundingBox,
) -> xr.Dataset:
    """Clip an xarray Dataset to the given spatial bounding box."""
    lats = ds.lat.values
    lons = ds.lon.values

    # pad by half a cell so we include cells whose edges overlap the bounds
    half_dy = abs(float(lats[1] - lats[0])) / 2
    half_dx = abs(float(lons[1] - lons[0])) / 2

    # Account for potential reversed latitude indexing (North to South vs South to North)
    if lats[0] > lats[-1]:
        lat_slice = slice(bounds.north + half_dy, bounds.south - half_dy)
    else:
        lat_slice = slice(bounds.south - half_dy, bounds.north + half_dy)

    ds = ds.sel(
        lat=lat_slice,
        lon=slice(bounds.west - half_dx, bounds.east + half_dx),
    )
    return ds


def _write_to_cache(ds: xr.Dataset, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": 9, "shuffle": True}
        for var in ds.data_vars
    }
    ds.to_netcdf(cache_path, encoding=encoding)
    logger.debug("Cached AORC data to %s", cache_path)
    return cache_path


def _read_from_cache(cache_path: Path) -> xr.Dataset:
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    return xr.open_dataset(cache_path)