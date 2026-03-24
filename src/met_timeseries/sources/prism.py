"""
PRISM climate data source.

The public API is :func:`fetch_prism_daily`, which downloads daily PRISM
rasters for a given bounding box and date range, and returns an
:class:`xarray.Dataset` with ``(time, lat, lon)`` dimensions.

PRISM data are downloaded from the Oregon State PRISM Climate Group HTTP
server (https://prism.oregonstate.edu).  The 4-km (``4kmD2``) daily product
is used.
"""

from __future__ import annotations

import datetime as dt
import logging
import tempfile
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

_URL_TEMPLATE = (
    "https://services.nacse.org/prism/data/public/4km/{variable}/{year}{month:02d}{day:02d}"
)


def fetch_prism_daily(
    bounds: BoundingBox,
    start: str,
    end: str | None = None,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Fetch daily PRISM climate data for the given bounding box and date range.

    Downloads the 4-km daily PRISM rasters for each variable and returns a
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

    Returns
    -------
    xarray.Dataset
        Daily Dataset with ``time``, ``lat``, and ``lon`` dimensions.
        The ``time`` coordinate carries proper ``datetime64`` timestamps.

    Raises
    ------
    ValueError
        If *end* is before *start*.
    RuntimeError
        If a download fails for any requested day.
    """
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
            da = _fetch_single_day(bounds, date, var)
            arrays_by_var[var].append(da)

    time_coords = [
        dt.datetime(d.year, d.month, d.day) for d in all_days
    ]

    dataset_vars: dict[str, xr.DataArray] = {}
    for var, day_arrays in arrays_by_var.items():
        stacked = xr.concat(day_arrays, dim="time")
        stacked["time"] = ("time", time_coords)
        dataset_vars[var] = stacked

    return xr.Dataset(dataset_vars)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_single_day(
    bounds: BoundingBox,
    date: dt.date,
    variable: str,
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

    Returns
    -------
    xarray.DataArray
        2-D DataArray with ``lat`` and ``lon`` dimensions (no ``time`` dim).
    """
    url = _URL_TEMPLATE.format(
        variable=variable, year=date.year, month=date.month, day=date.day
    )
    logger.debug("Fetching PRISM daily %s for %s from %s", variable, date, url)

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = _download_prism_zip(url, Path(tmpdir))
        bil_path = _extract_bil(zip_path)

        import rioxarray  # type: ignore[import]

        da = xr.open_dataarray(bil_path, engine="rasterio")
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


def _download_prism_zip(url: str, dest_dir: Path) -> Path:
    """Download a PRISM zip file to *dest_dir*; return path to the local copy.

    Parameters
    ----------
    url:
        Full URL to the PRISM zip file.
    dest_dir:
        Directory in which to save the downloaded zip.

    Returns
    -------
    Path
        Path to the downloaded zip file.

    Raises
    ------
    RuntimeError
        If the download fails.
    """
    dest = dest_dir / "prism_data.zip"
    logger.debug("Downloading %s -> %s", url, dest)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except Exception as exc:
        raise RuntimeError(f"Failed to download PRISM data from {url}: {exc}") from exc
    return dest


def _extract_bil(zip_path: Path) -> Path:
    """Extract the ``.bil`` file from *zip_path*; return its path.

    Parameters
    ----------
    zip_path:
        Path to the downloaded PRISM zip archive.

    Returns
    -------
    Path
        Path to the extracted ``.bil`` raster file.

    Raises
    ------
    RuntimeError
        If no ``.bil`` file is found inside the archive.
    """
    out_dir = zip_path.parent / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    bil_files = list(out_dir.glob("*.bil"))
    if not bil_files:
        raise RuntimeError(f"No .bil file found in {zip_path}")
    return bil_files[0]
