"""
PRISM climate data source.

The public API is :func:`fetch_prism`, which downloads monthly PRISM rasters
for a given bounding box and returns an :class:`xarray.Dataset`.

PRISM data are downloaded from the Oregon State PRISM Climate Group FTP/HTTP
server (https://prism.oregonstate.edu).  The 4-km (``4kmM3``) monthly product
is used by default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from met_timeseries.sources.base import BoundingBox

logger = logging.getLogger(__name__)

#: PRISM variable short-names available for the monthly 4-km product
AVAILABLE_VARIABLES: list[str] = [
    "ppt",   # precipitation (mm)
    "tmean", # mean temperature (°C)
    "tmax",  # maximum temperature (°C)
    "tmin",  # minimum temperature (°C)
    "tdmean",# mean dew-point temperature (°C)
    "vpdmin",# minimum vapour-pressure deficit (hPa)
    "vpdmax",# maximum vapour-pressure deficit (hPa)
]

_URL_TEMPLATE = (
    "https://services.nacse.org/prism/data/public/4km/{variable}/{year}{month:02d}"
)


def fetch_prism(
    bounds: BoundingBox,
    year: int,
    month: int,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
) -> xr.Dataset:
    """Fetch monthly PRISM climate data for the given bounding box.

    Downloads the 4-km monthly PRISM rasters for each variable and returns a
    spatially-subsetted :class:`xarray.Dataset`.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    year:
        Calendar year (e.g. 2010).
    month:
        Calendar month (1–12).
    variables:
        PRISM variable short-names to download.  Defaults to
        ``["ppt", "tmax", "tmin"]``.
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
        If data cannot be downloaded for the requested period.
    """
    if variables is None:
        variables = ["ppt", "tmax", "tmin"]

    logger.info("Fetching PRISM data: year=%d month=%d bounds=%r", year, month, bounds)

    arrays: dict[str, xr.DataArray] = {}
    for var in variables:
        da = _fetch_prism_variable(bounds, year, month, var, cache_dir)
        arrays[var] = da

    return xr.Dataset(arrays)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_prism_variable(
    bounds: BoundingBox,
    year: int,
    month: int,
    variable: str,
    cache_dir: str | None,
) -> xr.DataArray:
    """Download one PRISM monthly raster and subset to *bounds*."""
    cache_path = _cache_path(cache_dir, variable, year, month) if cache_dir else None

    if cache_path is not None and cache_path.exists():
        ds = xr.open_dataset(cache_path)
        return ds[variable]

    url = _URL_TEMPLATE.format(variable=variable, year=year, month=month)
    local_zip = _download_prism_zip(url, variable, year, month, cache_dir)
    bil_path = _extract_bil(local_zip)

    import rioxarray  # type: ignore[import]

    da = xr.open_dataarray(bil_path, engine="rasterio")
    da = da.rio.clip_box(
        minx=bounds.west, miny=bounds.south, maxx=bounds.east, maxy=bounds.north
    )
    da.name = variable

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        da.to_netcdf(cache_path)

    return da


def _cache_path(cache_dir: str, variable: str, year: int, month: int) -> Path:
    return Path(cache_dir) / "prism" / f"{variable}_{year}{month:02d}.nc"


def _download_prism_zip(
    url: str,
    variable: str,
    year: int,
    month: int,
    cache_dir: str | None,
) -> Path:
    """Download a PRISM zip file; return path to the local copy."""
    import urllib.request

    dest_dir = Path(cache_dir) / "prism" / "zips" if cache_dir else Path("/tmp/prism")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{variable}_{year}{month:02d}.zip"

    if not dest.exists():
        logger.debug("Downloading %s -> %s", url, dest)
        urllib.request.urlretrieve(url, dest)  # noqa: S310

    return dest


def _extract_bil(zip_path: Path) -> Path:
    """Extract a PRISM ``.bil`` file from *zip_path*; return its path."""
    import zipfile

    out_dir = zip_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    bil_files = list(out_dir.glob("*.bil"))
    if not bil_files:
        raise RuntimeError(f"No .bil file found in {zip_path}")
    return bil_files[0]
