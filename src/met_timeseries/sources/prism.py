"""
PRISM climate data source.

The public API is :func:`fetch_prism`, which downloads daily PRISM
rasters for a given bounding box and date range, and returns an
:class:`xarray.Dataset` with ``(time, lat, lon)`` dimensions.

PRISM data are downloaded from the Oregon State PRISM Climate Group HTTP
server (https://services.nacse.org/prism/data/get/).  Both 4-km and 800-m
daily products are supported.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
import time
from typing import List
import urllib.request
import zipfile
from pathlib import Path

import xarray as xr

from met_timeseries.sources.base import CACHE_BOUNDS, BoundingBox

logger = logging.getLogger(__name__)

#: PRISM variable short-names available for the daily 4-km product
AVAILABLE_VARIABLES: list[str] = [
    "ppt",    # precipitation (mm)
    "tmax",   # maximum temperature (°C)
    "tmin",   # minimum temperature (°C)
    "tmean",  # mean temperature (°C)
    "tdmean", # mean dew-point temperature (°C)
    "vpdmin", # minimum vapour-pressure deficit (hPa)
    "vpdmax", # maximum vapour-pressure deficit (hPa)
]



_URL_TEMPLATE = "https://services.nacse.org/prism/data/get/us/{resolution}/{variable}/{date}"
_RELEASE_DATE_URL_TEMPLATE = "https://services.nacse.org/prism/data/get/releaseDate/us/{resolution}/{variable}/{date}"

VALID_RESOLUTIONS = ("800m", "4km")


def fetch_prism(
    date: str,
    cache_dir: Path | str,
    bounds: BoundingBox | None = None,
    variables : list[str] = ['ppt', 'tmax', 'tmin'],
    resolution: str = "4km",
) -> xr.Dataset:
    """Fetch daily PRISM climate data for the given bounding box and date range.

    Downloads PRISM rasters for each variable and returns a
    spatially-subsetted :class:`xarray.Dataset`.

    Parameters
    ----------
    date:
        Date in ``"YYYY-MM-DD"`` format.
    variables:
        PRISM variable short-names to download.  Defaults to
        ``["ppt", "tmax", "tmin"]``.
    resolution:
        Grid resolution.  One of ``"800m"`` or ``"4km"``.
    cache_dir:
        Directory in which to persist downloaded zip files.  Each zip is saved as
        ``prism_{variable}_{resolution}_{date}.zip`` under that directory (created automatically if it does not exist).

    Returns
    -------
    xarray.Dataset
        Daily Dataset with ``time``, ``lat``, and ``lon`` dimensions.
        The ``time`` coordinate carries proper ``datetime64`` timestamps.

    Raises
    ------
    ValueError
        If *end* is before *start* or *resolution* is invalid.
    RuntimeError
        If a download fails for any requested day.
    """
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid resolution {resolution!r}; must be one of {VALID_RESOLUTIONS}"
        )

    if bounds is None:
        bounds = CACHE_BOUNDS

    # Validate date inputs and construct list of all dates to fetch
    date = dt.date.fromisoformat(date)
    logger.info(
        "Fetching PRISM daily data: date=%s variables=%s",
        date, variables,
    )


    # Download each day's raster, clip to bounds, and collect into lists by variable
    cache_path = Path(cache_dir) / f"prism_{resolution}_{date.strftime('%Y%m%d')}.nc"

    if cache_path.exists():
        ds = _load_from_cache(date,resolution,cache_dir)
        missing_vars = [var for var in variables if var not in ds.data_vars]
        if missing_vars:
            ds_missing = download(date = date,
                                  resolution = resolution, 
                                  variables = missing_vars, 
                                  cache_dir = cache_dir)
            ds = xr.merge([ds, ds_missing])
            _cache_dataset(ds,cache_path)
    else:
        ds = download(date = date,
                      resolution = resolution,
                      variables = variables,
                      cache_dir = cache_dir)
        _cache_dataset(ds,cache_path)

    ds = _clip_dataset(ds, bounds=bounds)
    return ds

def download(
    date: dt.date,
    resolution: str,
    cache_dir: Path | str | None = None,
    variables: List[str] = ['ppt', 'tmax', 'tmin']
) -> xr.DataArray:
    """Download one day's PRISM raster to  a zip file and return zip file path.

    Parameters
    ----------
    date:
        The calendar date to fetch.
    variable:
        PRISM variable short-name.
    resolution:
        Grid resolution (``"800m"`` or ``"4km"``).
    cache_dir:
        directory in which to persist the downloaded zip file since the web service must be written to disk regardless

    Returns
    -------
    xarray.DataArray
        2-D DataArray with ``lat`` and ``lon`` dimensions (no ``time`` dim).
    """

    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "prism_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


    arrays_by_var: dict[str, xr.DataArray] = {}
    for var in variables:
        url = _construct_url(var, resolution, date)
        zip_path = _download_url(
            url,
            cache_dir,
            variable=var,
            resolution=resolution,
            date= date,)
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")
        da = _load_from_zip(zip_path, var, resolution, date)
        zip_path.unlink()
        arrays_by_var[var] = da
        time.sleep(2)

    ds = _clip_dataset(xr.Dataset(arrays_by_var), CACHE_BOUNDS)
    # Add time dimension
    ds = ds.expand_dims(time=[dt.datetime(date.year, date.month, date.day)])
    return ds

def get_release_date(
    variable: str,
    date: str,
    resolution: str = "4km",
) -> dict:
    """Query the PRISM Release Date web service.

    Parameters
    ----------
    variable:
        PRISM variable short-name (must be in AVAILABLE_VARIABLES).
    date:
        Date in ``"YYYY-MM-DD"`` format.
    resolution:
        Grid resolution. One of ``"800m"`` or ``"4km"``.

    Returns
    -------
    dict
        Keys: ``data_date``, ``release_date``, ``element``, ``grid_count``, ``data_url``.

    Raises
    ------
    ValueError
        If *resolution* or *variable* is invalid.
    RuntimeError
        If the HTTP request fails.
    """
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid resolution {resolution!r}; must be one of {VALID_RESOLUTIONS}"
        )
    if variable not in AVAILABLE_VARIABLES:
        raise ValueError(
            f"Invalid variable {variable!r}; must be one of {AVAILABLE_VARIABLES}"
        )

    date_str = date.replace("-", "")
    url = (
        _RELEASE_DATE_URL_TEMPLATE.format(
            resolution=resolution, variable=variable, date=date_str
        )
        + "?json=true"
    )

    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            body = resp.read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch PRISM release date from {url}: {exc}"
        ) from exc

    return json.loads(body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

RESOLUTION_MAP = {
    "400m": "15s",
    "800m": "30s",
    "4km": "25m",
}



def _clip_dataset(ds: xr.Dataset, bounds: BoundingBox) -> xr.Dataset:
    """Clip an xarray Dataset to the given spatial bounding box."""
    # PRISM lat is descending (north→south); slice must match index order
    if ds.lat.values[0] > ds.lat.values[-1]:
        lat_slice = slice(bounds.north, bounds.south)
    else:
        lat_slice = slice(bounds.south, bounds.north)

    ds = ds.sel(
        lat=lat_slice,
        lon=slice(bounds.west, bounds.east),
    )
    return ds.load()

# def _clip_dataset(ds: xr.Dataset,bounds: BoundingBox) -> xr.Dataset:
#     """
#     Clip an xarray Dataset to the given spatial bounding box.
#     old method.
#     """
#     ds = ds.sel(
#         lat=slice(bounds.south, bounds.north),
#         lon=slice(bounds.west, bounds.east),
#     )
#     ds = ds.load()
#     return ds


def _get_zip_path(date: dt.date, variable: str, resolution: str, cache_dir: Path | str) -> Path:
    """Construct the expected zip file path for a given PRISM variable, date, and resolution."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = f"prism_{variable}_{resolution}_{date.strftime('%Y%m%d')}.zip"
    zip_path = cache_dir / zip_filename
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    return zip_path

def _load_from_cache(date: dt.date, resolution: str, cache_dir: Path | str) -> xr.DataArray:
    """Load a PRISM variable from the local cache if it exists."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"prism_{resolution}_{date.strftime('%Y%m%d')}.nc"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    return xr.open_dataset(cache_path)

def _cache_dataset(ds: xr.Dataset, cache_path: Path) -> Path:
    logger.debug("Compressing PRISM daily dataset to %s", cache_path)
    encoding = {var: {"zlib": True, "complevel": 9, "shuffle": True} for var in ds.data_vars}
    ds.to_netcdf(cache_path, encoding=encoding)
    return cache_path

def _construct_url(variable: str, resolution: str, date: dt.date) -> str:
    date_str = date.strftime("%Y%m%d")
    url = _URL_TEMPLATE.format(
        resolution=resolution, variable=variable, date=date_str
    )
    return url

def _load_from_zip(zip_path: Path, variable: str, resolution: str, date: dt.date) -> xr.DataArray:
    # Uses rasterio via xarray's "rasterio" engine, which supports reading from
    # zip files via vsizip.  This avoids the overhead of extracting the .tif to disk first.

    _resolution = RESOLUTION_MAP[resolution]
    date_str = date.strftime("%Y%m%d")

    vsi_path = f"/vsizip/{zip_path.as_posix()}/prism_{variable}_us_{_resolution}_{date_str}.tif"
    import rioxarray  # type: ignore[import]

    with xr.open_dataarray(vsi_path, engine="rasterio") as da:


        # Squeeze out the band dimension if present (common with rasterio engine)
        if "band" in da.dims:
            da = da.sel(band=1, drop=True)

        # Rename y/x to lat/lon to match NLDAS convention
        rename_map = {}
        if "y" in da.dims:
            rename_map["y"] = "lat"
        if "x" in da.dims:
            rename_map["x"] = "lon"
        if rename_map:
            da = da.rename(rename_map)

        da.name = variable

        # Load into memory before the temp directory is cleaned up
        da = da.load()

    return da


def _download_url(
    url: str,
    cache_dir: Path | str,
    variable: str ,
    resolution: str ,
    date: dt.date
    
) -> Path:
    """Download a PRISM zip file; return path to the local copy.

    When *cache_dir* is provided the zip is saved there as
    ``prism_{variable}_{resolution}_{date}.zip`` (the directory is created if
    it does not exist).

    Parameters
    ----------
    url:
        Full URL to the PRISM zip file.
    variable:
        PRISM variable short-name; used to build the cached filename.
    resolution:
        Grid resolution string; used to build the cached filename.
    date:
        Date string in ``YYYYMMDD`` format; used to build the cached filename.
    cache_dir:
        Persistent cache directory.

    Returns
    -------
    Path
        Path to the downloaded zip file.

    Raises
    ------
    RuntimeError
        If the download fails.
    """

    logger.debug("Downloading PRISM daily %s for %s from %s", variable, date, url)
    date = date.strftime("%Y%m%d")
    assert resolution in RESOLUTION_MAP.keys(), f"Invalid resolution {resolution!r}"
    assert variable in AVAILABLE_VARIABLES, f"Invalid variable {variable!r}"

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    dest = cache_path / f"prism_{variable}_{resolution}_{date}.zip"

    if dest.exists():
        logger.debug("Using cached PRISM zip at %s", dest)
    else:
        logger.debug("Downloading %s -> %s", url, dest)
        try:
            urllib.request.urlretrieve(url, dest)  # noqa: S310
        except Exception as exc:
            raise RuntimeError(f"Failed to download PRISM data from {url}: {exc}") from exc
    return dest