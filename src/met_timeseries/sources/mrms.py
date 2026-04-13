"""
MRMS (Multi-Radar Multi-Sensor) Gauge-Corrected QPE data source.

The public API is :func:`fetch_mrms`, which downloads hourly MRMS
GaugeCorr_QPE_01H rasters for a given bounding box and date, and returns an
:class:`xarray.Dataset` with ``(time, lat, lon)`` dimensions and a
``precip_mm`` variable.

MRMS data are downloaded from the Iowa Environmental Mesonet (IEM) archive at
https://mtarchive.geol.iastate.edu/.  No authentication is required.
"""

from __future__ import annotations

import datetime as dt
import gzip
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box as shapely_box

from met_timeseries.sources.base import CACHE_BOUNDS, BoundingBox

logger = logging.getLogger(__name__)

#: MRMS GaugeCorr_QPE_01H native resolution in degrees (~1 km)
_MRMS_RESOLUTION: float = 0.01

#: Cache filename prefix
_CACHE_PREFIX = "MRMS_GaugeCorr_QPE_01H_"

#: URL template for IEM MRMS archive
_URL_TEMPLATE = (
    "https://mtarchive.geol.iastate.edu/{YYYY}/{MM}/{DD}"
    "/mrms/ncep/GaugeCorr_QPE_01H"
    "/GaugeCorr_QPE_01H_00.00_{YYYYMMDD}-{HH}0000.grib2.gz"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_mrms(
    date: str,
    cache_dir: str | Path,
    bounds: BoundingBox | None = None,
) -> xr.Dataset:
    """Fetch MRMS GaugeCorr_QPE_01H hourly precipitation for a single day.

    Downloads all 24 hourly GRIB2 files from the IEM MRMS archive, merges
    them into a single Dataset, clips to :data:`CACHE_BOUNDS`, and caches the
    result as a compressed NetCDF.  On subsequent calls the cached file is
    loaded instead of re-downloading.

    Parameters
    ----------
    date:
        Date in ``"YYYY-MM-DD"`` format.
    cache_dir:
        Directory where the daily NetCDF cache file will be stored.
    bounds:
        Optional spatial bounding box in EPSG:4326.  When ``None`` the full
        cached extent (``CACHE_BOUNDS``) is returned.

    Returns
    -------
    xarray.Dataset
        Hourly Dataset with ``time``, ``lat``, and ``lon`` dimensions and a
        ``precip_mm`` variable (millimetres).

    Raises
    ------
    RuntimeError
        If all 24 hourly downloads fail.
    """
    date_obj = dt.date.fromisoformat(date)
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{_CACHE_PREFIX}{date_obj.strftime('%Y%m%d')}.nc"

    if cache_path.exists():
        return _load_from_cache(cache_path, bounds=bounds)

    ds = _download_day(date_obj, cache_dir)
    _cache_dataset(ds, cache_path)

    if bounds is not None:
        ds = _clip_dataset(ds, bounds=bounds)
    return ds.load()


def generate_grid(
    west: float = -130.0,
    south: float = 20.0,
    east: float = -60.0,
    north: float = 55.0,
    resolution: float = _MRMS_RESOLUTION,
) -> gpd.GeoDataFrame:
    """Return the MRMS grid as a GeoDataFrame of cell polygons.

    Parameters
    ----------
    west, south, east, north:
        Grid extent in EPSG:4326 degrees.  Defaults to approximate MRMS CONUS
        domain.
    resolution:
        Grid cell size in degrees.  Defaults to ``0.01`` (~1 km).

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``lat``, ``lon``, ``row_id``, ``column_id``, ``geometry``.
        CRS is EPSG:4326.
    """
    half = resolution / 2.0
    lons = np.arange(west, east + half, resolution)
    lats = np.arange(south, north + half, resolution)

    records = []
    for row_id, lat in enumerate(lats):
        for col_id, lon in enumerate(lons):
            records.append(
                {
                    "lat": round(lat, 7),
                    "lon": round(lon, 7),
                    "row_id": row_id,
                    "column_id": col_id,
                    "geometry": shapely_box(
                        lon - half, lat - half, lon + half, lat + half
                    ),
                }
            )

    return gpd.GeoDataFrame(records, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _download_day(date: dt.date, cache_dir: Path) -> xr.Dataset:
    """Download all 24 hourly GRIB2 files for *date* and return a merged Dataset.

    Failed hours are skipped with a warning.  A :exc:`RuntimeError` is raised
    only when all 24 hours fail.

    Parameters
    ----------
    date:
        The calendar date to download.
    cache_dir:
        Passed through to :func:`_download_hour` for temp-file placement.

    Returns
    -------
    xarray.Dataset
        Dataset with 24 time steps (or fewer if some hours were skipped),
        clipped to :data:`CACHE_BOUNDS`.
    """
    date_str = date.strftime("%Y%m%d")
    hourly_datasets: list[xr.Dataset] = []

    for hour in range(24):
        url = _URL_TEMPLATE.format(
            YYYY=date.strftime("%Y"),
            MM=date.strftime("%m"),
            DD=date.strftime("%d"),
            YYYYMMDD=date_str,
            HH=f"{hour:02d}",
        )
        try:
            ds_hour = _download_hour(url, date, hour)
            hourly_datasets.append(ds_hour)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            logger.warning(
                "Skipping MRMS hour %02d for %s due to error: %s",
                hour,
                date_str,
                exc,
            )

    if not hourly_datasets:
        raise RuntimeError(
            f"All 24 hourly MRMS downloads failed for {date_str}"
        )

    ds = xr.concat(
        sorted(hourly_datasets, key=lambda d: d["time"].values[0]),
        dim="time",
    )
    return ds


def _download_hour(url: str, date: dt.date, hour: int) -> xr.Dataset:
    """Download and decode a single gzipped GRIB2 hourly MRMS file.

    Steps:
    1. Download the ``.grib2.gz`` file into a temporary directory.
    2. Gunzip it to a ``.grib2`` temp file.
    3. Open with ``xr.open_dataset(engine="cfgrib")``.
    4. Rename ``latitude``/``longitude`` to ``lat``/``lon``.
    5. Convert longitudes from 0–360 to −180–180.
    6. Rename the precipitation variable to ``precip_mm``.
    7. Add a proper ``time`` coordinate via ``expand_dims``.
    8. Clip to :data:`CACHE_BOUNDS` immediately.
    9. Load into memory; clean up temp files in a ``finally`` block.

    Parameters
    ----------
    url:
        Full URL to the gzipped GRIB2 file.
    date:
        Calendar date; used to construct the ``time`` coordinate.
    hour:
        Hour of the day (0–23); used to construct the ``time`` coordinate.

    Returns
    -------
    xarray.Dataset
        Single-timestep Dataset clipped to :data:`CACHE_BOUNDS`.
    """
    import cfgrib  # noqa: F401 - ensure the engine is registered

    gz_path: Path | None = None
    grib_path: Path | None = None
    ds_raw: xr.Dataset | None = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gz_path = tmp / "mrms.grib2.gz"
            grib_path = tmp / "mrms.grib2"

            # --- download ---
            logger.debug("Downloading MRMS hour %02d from %s", hour, url)
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                gz_path.write_bytes(resp.read())

            # --- gunzip ---
            with gzip.open(gz_path, "rb") as f_in, open(grib_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            # --- open ---
            ds_raw = xr.open_dataset(
                str(grib_path),
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )

            # --- rename spatial coords ---
            rename_map: dict[str, str] = {}
            if "latitude" in ds_raw.coords:
                rename_map["latitude"] = "lat"
            if "longitude" in ds_raw.coords:
                rename_map["longitude"] = "lon"
            if rename_map:
                ds_raw = ds_raw.rename(rename_map)

            # --- 0–360 → −180–180 ---
            if "lon" in ds_raw.coords and float(ds_raw["lon"].max()) > 180.0:
                ds_raw = ds_raw.assign_coords(
                    lon=(ds_raw["lon"] + 180.0) % 360.0 - 180.0
                )
                ds_raw = ds_raw.sortby("lon")

            # --- rename precip variable ---
            precip_candidates = [
                v for v in ds_raw.data_vars
                if v in ("unknown", "paramId_0", "tp", "APCP")
                or "precip" in v.lower()
                or "rain" in v.lower()
            ]
            if not precip_candidates:
                # fall back to first data variable
                precip_candidates = list(ds_raw.data_vars)
            if not precip_candidates:
                raise ValueError(f"No data variables found in MRMS GRIB2 at {url}")
            ds_raw = ds_raw.rename({precip_candidates[0]: "precip_mm"})

            # --- add time coordinate ---
            timestamp = dt.datetime(date.year, date.month, date.day, hour)
            ds_raw = ds_raw.expand_dims(
                time=[np.datetime64(timestamp, "ns")]
            )

            # --- drop any non-spatial, non-time coords that cfgrib adds ---
            coords_to_drop = [
                c for c in ds_raw.coords
                if c not in ("lat", "lon", "time")
                and c not in ds_raw.dims
            ]
            if coords_to_drop:
                ds_raw = ds_raw.drop_vars(coords_to_drop)

            # --- clip to CACHE_BOUNDS immediately ---
            ds_clipped = _clip_dataset(ds_raw, bounds=CACHE_BOUNDS)
            result = ds_clipped.load()

    finally:
        if ds_raw is not None:
            try:
                ds_raw.close()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to close raw GRIB2 dataset", exc_info=True)

    return result


def _clip_dataset(ds: xr.Dataset, bounds: BoundingBox) -> xr.Dataset:
    """Clip *ds* to *bounds* with a half-cell pad.

    Works for both ascending and descending ``lat`` arrays.

    Parameters
    ----------
    ds:
        Dataset with ``lat`` and ``lon`` dimensions.
    bounds:
        Spatial bounding box.

    Returns
    -------
    xarray.Dataset
        Subset of *ds* whose cells overlap *bounds*.
    """
    lats = ds.lat.values
    lons = ds.lon.values

    # pad by half a cell so cells whose edges overlap the bounds are included
    half_dy = abs(float(lats[1] - lats[0])) / 2
    half_dx = abs(float(lons[1] - lons[0])) / 2

    if lats[0] > lats[-1]:
        # descending lat (north → south)
        lat_slice = slice(bounds.north + half_dy, bounds.south - half_dy)
    else:
        # ascending lat (south → north)
        lat_slice = slice(bounds.south - half_dy, bounds.north + half_dy)

    ds = ds.sel(
        lat=lat_slice,
        lon=slice(bounds.west - half_dx, bounds.east + half_dx),
    )
    return ds


def _cache_dataset(ds: xr.Dataset, cache_path: Path) -> Path:
    """Write *ds* to *cache_path* as a compressed NetCDF-4 file.

    Parameters
    ----------
    ds:
        Dataset to cache.
    cache_path:
        Destination path.

    Returns
    -------
    Path
        Path to the written file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": 9, "shuffle": True}
        for var in ds.data_vars
    }
    ds.to_netcdf(cache_path, encoding=encoding)
    logger.debug("Cached MRMS data to %s", cache_path)
    return cache_path


def _load_from_cache(
    cache_path: Path, bounds: BoundingBox | None = None
) -> xr.Dataset:
    """Load an MRMS daily Dataset from a cached NetCDF file.

    Parameters
    ----------
    cache_path:
        Path to the cached ``.nc`` file.
    bounds:
        Optional clip bounds.  Defaults to :data:`CACHE_BOUNDS`.

    Returns
    -------
    xarray.Dataset
        In-memory Dataset clipped to *bounds*.

    Raises
    ------
    FileNotFoundError
        If *cache_path* does not exist.
    """
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    if bounds is None:
        bounds = CACHE_BOUNDS

    ds = xr.open_dataset(cache_path)
    ds_clipped = _clip_dataset(ds, bounds=bounds)
    ds.close()
    return ds_clipped.load()
