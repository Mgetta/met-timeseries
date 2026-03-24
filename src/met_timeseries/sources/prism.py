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
    bounds: BoundingBox,
    start: str,
    end: str | None = None,
    variables: list[str] | None = None,
    resolution: str = "4km",
    cache_dir: Path | str | None = None,
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
        Optional directory in which to persist downloaded zip files.  When
        ``None`` (the default), zip files are written to a temporary directory
        and discarded after extraction.  When a path is provided, each zip is
        saved as ``prism_{variable}_{resolution}_{date}.zip`` under that
        directory (created automatically if it does not exist).

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

    if variables is None:
        variables = ["ppt", "tmax", "tmin"]

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

    arrays_by_var: dict[str, list[xr.DataArray]] = {var: [] for var in variables}

    for date in all_days:
        for var in variables:
            da = _fetch_single_day(bounds, date, var, resolution, cache_dir=cache_dir)
            arrays_by_var[var].append(da)
        time.sleep(2)

    time_coords = [
        dt.datetime(d.year, d.month, d.day) for d in all_days
    ]

    dataset_vars: dict[str, xr.DataArray] = {}
    for var, day_arrays in arrays_by_var.items():
        stacked = xr.concat(day_arrays, dim="time")
        stacked["time"] = ("time", time_coords)
        dataset_vars[var] = stacked

    return xr.Dataset(dataset_vars)


# Keep old name as an alias for backwards compatibility
fetch_prism_daily = fetch_prism


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

def _fetch_single_day(
    bounds: BoundingBox,
    date: dt.date,
    variable: str,
    resolution: str = "4km",
    cache_dir: Path | str | None = None,
) -> xr.DataArray:
    """Download one day's PRISM raster, clip to *bounds*, and return a DataArray.

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
        Optional directory in which to persist the downloaded zip file.
        When ``None``, the zip is discarded after extraction.

    Returns
    -------
    xarray.DataArray
        2-D DataArray with ``lat`` and ``lon`` dimensions (no ``time`` dim).
    """
    date_str = date.strftime("%Y%m%d")
    url = _URL_TEMPLATE.format(
        resolution=resolution, variable=variable, date=date_str
    )
    logger.debug("Fetching PRISM daily %s for %s from %s", variable, date, url)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        zip_path = _download_prism_zip(
            url,
            tmp_path,
            variable=variable,
            resolution=resolution,
            date=date_str,
            cache_dir=cache_dir,
        )
        tif_path = _extract_tif(zip_path, extract_dir=tmp_path)

        import rioxarray  # type: ignore[import]

        da = xr.open_dataarray(tif_path, engine="rasterio")
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


def _download_prism_zip(
    url: str,
    dest_dir: Path,
    variable: str = "",
    resolution: str = "",
    date: str = "",
    cache_dir: Path | str | None = None,
) -> Path:
    """Download a PRISM zip file; return path to the local copy.

    When *cache_dir* is provided the zip is saved there as
    ``prism_{variable}_{resolution}_{date}.zip`` (the directory is created if
    it does not exist).  When *cache_dir* is ``None`` the zip is written to
    *dest_dir* as ``prism_data.zip`` (existing behaviour).

    Parameters
    ----------
    url:
        Full URL to the PRISM zip file.
    dest_dir:
        Fallback directory used when *cache_dir* is ``None``.
    variable:
        PRISM variable short-name; used to build the cached filename.
    resolution:
        Grid resolution string; used to build the cached filename.
    date:
        Date string in ``YYYYMMDD`` format; used to build the cached filename.
    cache_dir:
        Optional persistent cache directory.

    Returns
    -------
    Path
        Path to the downloaded zip file.

    Raises
    ------
    RuntimeError
        If the download fails.
    """
    if cache_dir is not None:
        if not variable or not resolution or not date:
            raise ValueError(
                "variable, resolution, and date are required when cache_dir is provided"
            )
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        dest = cache_path / f"prism_{variable}_{resolution}_{date}.zip"
    else:
        dest = dest_dir / "prism_data.zip"
    logger.debug("Downloading %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except Exception as exc:
        raise RuntimeError(f"Failed to download PRISM data from {url}: {exc}") from exc
    return dest


def _extract_tif(zip_path: Path, extract_dir: Path | None = None) -> Path:
    """Extract the ``.tif`` file from *zip_path*; return its path.

    Parameters
    ----------
    zip_path:
        Path to the downloaded PRISM zip archive.
    extract_dir:
        Directory in which to extract the archive.  Defaults to the parent
        directory of *zip_path* when ``None``.

    Returns
    -------
    Path
        Path to the extracted ``.tif`` raster file.

    Raises
    ------
    RuntimeError
        If no ``.tif`` file is found inside the archive.
    """
    out_dir = (extract_dir if extract_dir is not None else zip_path.parent) / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    tif_files = list(out_dir.glob("*.tif"))
    if not tif_files:
        raise RuntimeError(f"No .tif file found in {zip_path}")
    return tif_files[0]
