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
import urllib.request
import zipfile
from pathlib import Path

import xarray as xr

from met_timeseries.sources.base import BoundingBox

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
    start: str,
    cache_dir: Path | str,
    variables : list[str],
    bounds: BoundingBox | None = None,
    end: str | None = None,
    resolution: str = "4km",
) -> xr.Dataset:
    """Fetch daily PRISM climate data for the given bounding box and date range.

    Downloads PRISM rasters for each variable and returns a
    spatially-subsetted :class:`xarray.Dataset`.

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

    # Validate date inputs and construct list of all dates to fetch
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end) if end is not None else start_date
    if end_date < start_date:
        raise ValueError(
            f"end ({end!r}) must not be before start ({start!r})"
        )
    logger.info(
        "Fetching PRISM daily data: start=%s end=%s variables=%s bounds=%r",
        start_date, end_date, variables, bounds,
    )

    all_days = [
        start_date + dt.timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]

    time_coords = [
        dt.datetime(d.year, d.month, d.day) for d in all_days
    ]

    # Download each day's raster, clip to bounds, and collect into lists by variable
    arrays_by_var: dict[str, list[xr.DataArray]] = {var: [] for var in variables}
    for date in all_days:
        for var in variables:
            try:
                da = _load_from_cache(date, var, resolution, cache_dir)
            except FileNotFoundError:
                cache_path = download(date, var, resolution, cache_dir,compress=True,bounds = bounds)
                da = _load_from_cache(date, var, resolution, cache_dir)
            arrays_by_var[var].append(da)
            time.sleep(2)

    # Stack each variable's daily arrays along a new time dimension, and combine into a Dataset
    dataset_vars: dict[str, xr.DataArray] = {}
    for var, day_arrays in arrays_by_var.items():
        stacked = xr.concat(day_arrays, dim="time")
        stacked["time"] = ("time", time_coords)
        dataset_vars[var] = stacked

    return xr.Dataset(dataset_vars)

def _get_zip_path(date: dt.date, variable: str, resolution: str, cache_dir: Path | str) -> Path:
    """Construct the expected zip file path for a given PRISM variable, date, and resolution."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = f"prism_{variable}_{resolution}_{date.strftime('%Y%m%d')}.zip"
    zip_path = cache_dir / zip_filename
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    return zip_path

def _load_from_cache(date: dt.date, variable: str, resolution: str, cache_dir: Path | str) -> xr.DataArray:
    """Load a PRISM variable from the local cache if it exists."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = f"prism_{variable}_{resolution}_{date.strftime('%Y%m%d')}.nc"
    nc_path = cache_dir / zip_filename
    if not nc_path.exists():
        raise FileNotFoundError(f"Zip file not found: {nc_path}")
    return xr.open_dataset(nc_path)[variable]



def _compress(zip_path: Path | str, variable: str,resolution: str,date: dt.date,bounds: BoundingBox | None = None) -> Path:

    zip_path = Path(zip_path)
    compress_path = zip_path.with_suffix(".nc")
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    da = _load_from_zip(zip_path, variable, resolution, date,bounds)
    encoding = {
        da.name: {"zlib": True, "complevel": 9,"shuffle": True}
    }

    da.to_netcdf(compress_path, encoding=encoding)
    logger.debug("Cached NLDAS data to %s", compress_path)


def download(
    date: dt.date,
    variable: str,
    resolution: str,
    cache_dir: Path | str,
    compress: bool = True,
    bounds: BoundingBox | None = None
) -> xr.DataArray:
    """Download one day's PRISM raster to  a zip file and return zip file path.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
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

    url = _construct_url(variable, resolution, date)
    zip_path = _download_url(
        url,
        cache_dir,
        variable=variable,
        resolution=resolution,
        date= date,
    )
    if compress:
        logger.debug("Compressing PRISM daily grid %s for %s", variable, date)
        output_path = _compress(zip_path, variable, resolution, date,bounds)
        # delete the original zip file after compression
        logger.debug("Deleting original PRISM zip file %s", zip_path)
        #zip_path.unlink()
    else:
        output_path = zip_path
    return output_path

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

def _construct_url(variable: str, resolution: str, date: dt.date) -> str:
    date_str = date.strftime("%Y%m%d")
    url = _URL_TEMPLATE.format(
        resolution=resolution, variable=variable, date=date_str
    )
    return url

def _load_from_zip(zip_path: Path, variable: str, resolution: str, date: dt.date, bounds: BoundingBox = None) -> xr.DataArray:
    # Uses rasterio via xarray's "rasterio" engine, which supports reading from
    # zip files via vsizip.  This avoids the overhead of extracting the .tif to disk first.

    _resolution = RESOLUTION_MAP[resolution]
    date_str = date.strftime("%Y%m%d")

    vsi_path = f"/vsizip/{zip_path.as_posix()}/prism_{variable}_us_{_resolution}_{date_str}.tif"
    import rioxarray  # type: ignore[import]

    da = xr.open_dataarray(vsi_path, engine="rasterio")

    if bounds is not None:
        da = da.rio.clip_box(
            minx=bounds.west, miny=bounds.south, maxx=bounds.east, maxy=bounds.north
        )

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