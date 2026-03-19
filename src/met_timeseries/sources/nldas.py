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

import calendar
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

_GIOVANNI_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"

#: Number of metadata header lines in a Giovanni Time Series CSV response.
_GIOVANNI_HEADER_LINES = 13

#: Mapping from our short variable names to Giovanni Time Series Service
#: data parameter identifiers for NLDAS_FORA0125_H v2.0.
#: The naming pattern is: {collection_with_underscores}_{giovanni_short_name}
_GIOVANNI_VARIABLES: dict[str, str] = {
    "TMP": "NLDAS_FORA0125_H_2_0_Tair",
    "SPFH": "NLDAS_FORA0125_H_2_0_Qair",
    "PRES": "NLDAS_FORA0125_H_2_0_PSurf",
    "UGRD": "NLDAS_FORA0125_H_2_0_Wind_E",
    "VGRD": "NLDAS_FORA0125_H_2_0_Wind_N",
    "DLWRF": "NLDAS_FORA0125_H_2_0_LWdown",
    "PEVAP": "NLDAS_FORA0125_H_2_0_PotEvap",
    "APCP": "NLDAS_FORA0125_H_2_0_Rainf",
    "DSWRF": "NLDAS_FORA0125_H_2_0_SWdown",
}

#: NLDAS-2 grid resolution in degrees; used as the cell buffer when selecting
#: grid cells that fall within a bounding box.
_NLDAS_RESOLUTION: float = 0.125


def _bounds_to_polygon(bounds: BoundingBox):
    """Return a Shapely box polygon for *bounds*."""
    return box(bounds.west, bounds.south, bounds.east, bounds.north)


def get_nldas_gridcells(bounds: BoundingBox) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of NLDAS grid cells that intersect the bounding box."""
    grid = generate_nldas_grid(
        west=bounds.west - _NLDAS_RESOLUTION,
        south=bounds.south - _NLDAS_RESOLUTION,
        east=bounds.east + _NLDAS_RESOLUTION,
        north=bounds.north + _NLDAS_RESOLUTION,
    )
    bbox_polygon = _bounds_to_polygon(bounds)
    grid_in_bounds = grid[grid.intersects(bbox_polygon)]
    return grid_in_bounds[["lat_center", "lon_center", "geometry"]].drop_duplicates()

def download_datarods(
    bounds: BoundingBox,
    year: int = 1995,
    month: int | None = None,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
    end_year: int | None = None,
) -> dict[tuple[float, float], dict[str, pd.Series]]:
    """Download raw NLDAS-2 per-cell timeseries via Giovanni Time Series API.

    This function is independent of any weighting.  It determines which
    NLDAS grid cells fall within *bounds*, authenticates with NASA Earthdata
    once, then fetches hourly timeseries for each cell.  Raw per-cell data
    is optionally cached as CSV files.

    When *month* is ``None`` the data is requested in yearly chunks
    (one API call per year) rather than monthly, which greatly reduces
    the total number of HTTP requests.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326 used to select grid cells.
    year:
        Start calendar year when *month* is ``None``, or the specific
        calendar year when *month* is given.  Defaults to ``1995``.
    month:
        Calendar month (1–12).  When ``None`` (the default), full years
        between *year* and *end_year* are downloaded (one request per
        year) and the per-cell series are concatenated.
    variables:
        NLDAS-2 short variable names to fetch. Defaults to
        ``["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    cache_dir:
        Optional directory for per-cell CSV cache files.
    end_year:
        Last calendar year (inclusive) when downloading multiple months.
        Ignored when *month* is specified.  Defaults to the current year.

    Returns
    -------
    dict[tuple[float, float], dict[str, pandas.Series]]
        Mapping from ``(lat, lon)`` to a dict of variable name →
        hourly ``pd.Series`` with ``DatetimeIndex``.
    """
    import earthaccess

    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    # Build list of (start_date, end_date) pairs to fetch.
    # When a single month is specified we make one request for that month;
    # otherwise we request full years to minimise API calls.
    if month is not None:
        _, n_days = calendar.monthrange(year, month)
        date_ranges = [
            (f"{year}-{month:02d}-01T00:00:00",
             f"{year}-{month:02d}-{n_days:02d}T23:00:00"),
        ]
    else:
        if end_year is None:
            end_year = dt.datetime.now().year
        date_ranges = [
            (f"{y}-01-01T00:00:00", f"{y}-12-31T23:00:00")
            for y in range(year, end_year + 1)
        ]

    # Determine NLDAS grid cells that fall within the bounding box
    unique_cells = get_nldas_gridcells(bounds)
    
    # Authenticate once and reuse the token for all requests
    earthaccess.login()
    token = earthaccess.get_edl_token()["access_token"]

    n_cells = len(unique_cells)
    logger.info(
        "Downloading Giovanni timeseries: %d unique cells × %d variables for %d period(s)",
        n_cells, len(variables), len(date_ranges),
    )

    cell_data: dict[tuple[float, float], dict[str, pd.Series]] = {}
    for i, (_, row) in enumerate(unique_cells.iterrows(), start=1):
        lat, lon = row["lat_center"], row["lon_center"]
        logger.debug("Fetching cell %d/%d: lat=%s lon=%s", i, n_cells, lat, lon)
        cell_data[(lat, lon)] = {}
        for var in variables:
            chunk_series: list[pd.Series] = []
            for start_date, end_date in date_ranges:
                text = _cache_giovanni_response(
                    lat=lat, lon=lon, variable=var,
                    start_date=start_date, end_date=end_date,
                    cache_dir=cache_dir, _token=token,
                )
                chunk_series.append(_parse_giovanni_response(text))
            if len(chunk_series) == 1:
                cell_data[(lat, lon)][var] = chunk_series[0]
            else:
                cell_data[(lat, lon)][var] = pd.concat(chunk_series).sort_index()

    return cell_data


def compute_weighted_averages(
    cell_data: dict[tuple[float, float], dict[str, pd.Series]],
    weights: pd.DataFrame,
    variables: list[str] | None = None,
    polygon_id_col: str = "metzone_id",
) -> dict[str, pd.DataFrame]:
    """Compute weighted spatial averages from per-cell timeseries data.

    Takes raw per-cell data (as returned by :func:`download_datarods`) and
    a weight table, and produces one timeseries per polygon per variable.
    This function is independent of the download method.

    Parameters
    ----------
    cell_data:
        Raw per-cell data as returned by :func:`download_datarods`.
        Mapping from ``(lat, lon)`` to ``dict[variable_name, pd.Series]``.
    weights:
        Weight table with columns ``lat_center``, ``lon_center``,
        ``{polygon_id_col}``, and ``weight``.  Weights should sum
        to 1.0 per polygon (as returned by :func:`compute_nldas_weights`).
    variables:
        Variable names to process.  If ``None``, uses all variables
        found in the first cell of *cell_data*.
    polygon_id_col:
        Column name in *weights* that identifies each polygon.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from variable name to a DataFrame with a
        ``DatetimeIndex`` and one column per polygon.
    """
    if variables is None:
        first_key = next(iter(cell_data))
        variables = list(cell_data[first_key].keys())

    results: dict[str, pd.DataFrame] = {}
    for var in variables:
        polygon_series: dict = {}
        for polygon_id, grp in weights.groupby(polygon_id_col):
            first_key = (grp.iloc[0]["lat_center"], grp.iloc[0]["lon_center"])
            weighted_sum = pd.Series(
                0.0, index=cell_data[first_key][var].index
            )
            for _, wrow in grp.iterrows():
                key = (wrow["lat_center"], wrow["lon_center"])
                weighted_sum = weighted_sum.add(
                    cell_data[key][var] * wrow["weight"], fill_value=0
                )
            polygon_series[polygon_id] = weighted_sum
        results[var] = pd.DataFrame(polygon_series)

    return results


def process_nldas(
    bounds: BoundingBox,
    year: int = 1995,
    month: int | None = None,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
    weights: pd.DataFrame | None = None,
    polygon_id_col: str = "metzone_id",
    end_year: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Download NLDAS-2 data via datarods and compute weighted averages.

    This is the main entry point for the datarods-based workflow.  It
    downloads raw per-cell data with :func:`download_datarods`, then
    derives area-overlap weights automatically from the bounding box and
    the NLDAS grid (unless pre-computed *weights* are supplied), and
    finally applies :func:`compute_weighted_averages`.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    year:
        Start calendar year when *month* is ``None``, or the specific
        calendar year when *month* is given.  Defaults to ``1995``.
    month:
        Calendar month (1–12).  When ``None`` (the default), all months
        between *year* and *end_year* are downloaded.
    variables:
        NLDAS-2 short variable names to fetch.  Defaults to
        ``["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    cache_dir:
        Optional directory for caching raw per-cell CSV files.
    weights:
        Pre-computed weight table (see :func:`compute_nldas_weights`).
        If ``None`` (default), weights are derived automatically from
        *bounds* and the NLDAS grid.
    polygon_id_col:
        Column name in *weights* that identifies each polygon.  When
        weights are auto-derived the single polygon is labelled
        ``"bbox"``.
    end_year:
        Last calendar year (inclusive) when downloading multiple months.
        Ignored when *month* is specified.  Defaults to the current year.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from variable name to a DataFrame with a
        ``DatetimeIndex`` and one column per polygon.
    """
    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    cell_data = download_datarods(
        bounds=bounds,
        year=year,
        month=month,
        variables=variables,
        cache_dir=cache_dir,
        end_year=end_year,
    )

    if weights is None:
        bbox_polygon = _bounds_to_polygon(bounds)
        polygons_gdf = gpd.GeoDataFrame(
            {polygon_id_col: ["bbox"], "geometry": [bbox_polygon]},
            crs="EPSG:4326",
        )
        weights = compute_nldas_weights(
            polygons_gdf, polygon_id_col=polygon_id_col,
        )

    return compute_weighted_averages(
        cell_data=cell_data,
        weights=weights,
        variables=variables,
        polygon_id_col=polygon_id_col,
    )


def _open_and_subset_granule(
    file_obj: object,
    variables: list[str],
    bounds: BoundingBox,
) -> xr.Dataset:
    """Open a single NLDAS-2 granule file object and return a spatially subsetted Dataset.

    Parameters
    ----------
    file_obj:
        A file-like object (e.g. from :func:`earthaccess.open`) pointing to an
        NLDAS-2 hourly NetCDF granule.
    variables:
        Variable names to retain.
    bounds:
        Spatial bounding box; only data within these bounds is returned.

    Returns
    -------
    xarray.Dataset
        In-memory Dataset containing only *variables* and the spatial extent
        defined by *bounds*.
    """
    ds = xr.open_dataset(file_obj, engine="h5netcdf")
    ds = ds[variables]
    ds = ds.sel(
        lat=slice(bounds.south, bounds.north),
        lon=slice(bounds.west, bounds.east),
    )
    ds = ds.load()
    return ds


def _search_nldas_granules(
    bounds: BoundingBox,
    year: int,
    month: int,
) -> list:
    """Search for NLDAS-2 granules for a given month via CMR.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    year:
        Calendar year.
    month:
        Calendar month (1–12).

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

    _, n_days = calendar.monthrange(year, month)
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{n_days:02d}"

    results = earthaccess.search_data(
        short_name="NLDAS_FORA0125_H",
        version="2.0",
        temporal=(start, end),
        bounding_box=(bounds.west, bounds.south, bounds.east, bounds.north),
    )

    if not results:
        raise RuntimeError(f"No NLDAS-2 granules found for {year}-{month:02d}")

    logger.debug("Found %d NLDAS-2 granules for %d-%02d", len(results), year, month)
    return results


def _fetch_month_granules(
    bounds: BoundingBox,
    year: int,
    month: int,
    variables: list[str],
    max_connections: int,
) -> xr.Dataset:
    """Download and subset all granules for a single month, returning a Dataset.

    Opens granules concurrently using :class:`concurrent.futures.ThreadPoolExecutor`.
    Failed granules are skipped with a warning; a :exc:`RuntimeError` is raised
    only when *all* granules for the month fail.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    variables:
        Variable names to include.
    max_connections:
        Maximum concurrent granule downloads.

    Returns
    -------
    xarray.Dataset
        Monthly Dataset concatenated along the ``time`` dimension.
    """
    import earthaccess

    results = _search_nldas_granules(bounds, year, month)

    file_objs = earthaccess.open(results)

    granule_datasets: list[xr.Dataset] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_connections) as executor:
        futures = {
            executor.submit(_open_and_subset_granule, fobj, variables, bounds): i
            for i, fobj in enumerate(file_objs)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                granule_datasets.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping granule %d for %d-%02d due to error: %s",
                    idx, year, month, exc,
                )

    if not granule_datasets:
        raise RuntimeError(
            f"All granules failed for {year}-{month:02d}"
        )

    ds = xr.concat(
        sorted(granule_datasets, key=lambda d: d["time"].values[0]),
        dim="time",
    )

    # Some NLDAS-2 granules store time as a numeric offset rather than
    # datetime64. Reconstruct the coordinate from the known month start if
    # the decoded dtype is not already datetime64.
    if "time" not in ds.coords or not np.issubdtype(ds["time"].dtype, np.datetime64):
        n_hours = ds.sizes["time"]
        start_ts = np.datetime64(dt.datetime(year, month, 1), "ns")
        timestamps = [start_ts + np.timedelta64(h, "h") for h in range(n_hours)]
        ds = ds.assign_coords(time=timestamps)

    return ds


def fetch_nldas_grid(
    bounds: BoundingBox,
    year: int,
    month: int | None = None,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
    end_year: int | None = None,
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
    year:
        Start calendar year (e.g. 2010).
    month:
        Calendar month (1–12).  When ``None`` (the default), all months from
        ``(year, 1)`` through ``(end_year, 12)`` are fetched and concatenated.
    variables:
        NLDAS-2 short variable names to include.  Defaults to ``["APCP",
        "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    cache_dir:
        If provided, subsetted daily NetCDF files are cached here and
        loaded on subsequent calls for the same days.
    end_year:
        Last calendar year to include when ``month=None``.  Defaults to the
        current year when not specified.
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
    import earthaccess

    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    # Build list of (year, month) pairs to process
    if month is not None:
        ym_pairs = [(year, month)]
    else:
        if end_year is None:
            end_year = dt.datetime.now().year
        ym_pairs = [
            (y, m)
            for y in range(year, end_year + 1)
            for m in range(1, 13)
        ]

    logger.info(
        "Fetching NLDAS-2 data: %d month(s) starting %d-%02d bounds=%r",
        len(ym_pairs), ym_pairs[0][0], ym_pairs[0][1], bounds,
    )

    # Compute uncached days per month upfront to avoid duplicate filesystem checks
    # and to determine whether earthaccess authentication is needed.
    month_uncached_days: dict[tuple[int, int], set[int]] = {}
    _needs_download = False
    if cache_dir is not None:
        for y, m in ym_pairs:
            _, n_days = calendar.monthrange(y, m)
            uncached = {
                d for d in range(1, n_days + 1)
                if not _daily_cache_path(cache_dir, y, m, d).exists()
            }
            month_uncached_days[(y, m)] = uncached
            if uncached:
                _needs_download = True
    else:
        for y, m in ym_pairs:
            _, n_days = calendar.monthrange(y, m)
            month_uncached_days[(y, m)] = set(range(1, n_days + 1))
        _needs_download = True

    if _needs_download:
        earthaccess.login()

    monthly_datasets: list[xr.Dataset] = []
    for y, m in ym_pairs:
        _, n_days = calendar.monthrange(y, m)
        uncached_days = month_uncached_days[(y, m)]

        if not uncached_days:
            # All days cached — load from daily cache files
            logger.debug("Loading NLDAS-2 from daily cache for %d-%02d", y, m)
            day_datasets = [
                xr.open_dataset(_daily_cache_path(cache_dir, y, m, d))
                for d in range(1, n_days + 1)
            ]
            monthly_datasets.append(xr.concat(day_datasets, dim="time"))
            continue

        # Download full month's granules
        month_ds = _fetch_month_granules(bounds, y, m, variables, max_connections)

        # Group by day and cache uncached days
        times = pd.DatetimeIndex(month_ds.time.values)
        day_datasets = []
        for d in range(1, n_days + 1):
            if d not in uncached_days:
                # Load this day from cache
                day_datasets.append(
                    xr.open_dataset(_daily_cache_path(cache_dir, y, m, d))
                )
                continue

            day_indices = np.where(times.day == d)[0]
            if len(day_indices) == 0:
                logger.warning(
                    "No granules found for day %d of %d-%02d", d, y, m
                )
                continue

            day_ds = month_ds.isel(time=day_indices)

            if cache_dir is not None:
                cache_path = _daily_cache_path(cache_dir, y, m, d)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                day_ds.to_netcdf(cache_path)
                logger.debug("Cached NLDAS-2 day to: %s", cache_path)

            day_datasets.append(day_ds)

        if not day_datasets:
            raise RuntimeError(f"No usable granules found for any day in {y}-{m:02d}")

        monthly_datasets.append(xr.concat(day_datasets, dim="time"))

    if len(monthly_datasets) == 1:
        return monthly_datasets[0]

    return xr.concat(monthly_datasets, dim="time")


# ---------------------------------------------------------------------------
# Grid generation and weight computation
# ---------------------------------------------------------------------------

def generate_nldas_grid(
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


def compute_nldas_weights(
    polygons: gpd.GeoDataFrame,
    polygon_id_col: str = "metzone_id",
    nldas_grid: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Compute area-overlap weights between polygons and the NLDAS grid.

    For each polygon, finds all NLDAS grid cells that intersect it and
    computes the fractional overlap area. Weights are normalized so they
    sum to 1.0 per polygon.

    Parameters
    ----------
    polygons:
        GeoDataFrame of user polygons (e.g., dissolved metzones).
    polygon_id_col:
        Column name for the polygon identifier.
    nldas_grid:
        Pre-generated NLDAS grid GeoDataFrame. If None, generates one
        clipped to the total bounds of *polygons* (with a small buffer).

    Returns
    -------
    pandas.DataFrame
        Columns: ``[polygon_id_col, "lat_center", "lon_center", "weight"]``.
        Weights sum to 1.0 per polygon.
    """
    if nldas_grid is None:
        minx, miny, maxx, maxy = polygons.total_bounds
        buf = 0.25
        nldas_grid = generate_nldas_grid(
            west=minx - buf,
            south=miny - buf,
            east=maxx + buf,
            north=maxy + buf,
        )

    # Ensure matching CRS
    polygons = polygons.to_crs("EPSG:4326")
    nldas_grid = nldas_grid.to_crs("EPSG:4326")

    # Compute polygon areas in equal-area projection for accuracy
    polygons_ea = polygons.to_crs("EPSG:6933")
    poly_areas = polygons_ea.geometry.area

    # Intersection
    intersection = gpd.overlay(
        polygons[[polygon_id_col, "geometry"]],
        nldas_grid[["lat_center", "lon_center", "geometry"]],
        how="intersection",
    )

    # Compute overlap area in equal-area projection
    intersection_ea = intersection.to_crs("EPSG:6933")
    intersection["overlap_area"] = intersection_ea.geometry.area

    # Attach polygon area for normalization
    area_map = dict(zip(polygons[polygon_id_col], poly_areas))
    intersection["poly_area"] = intersection[polygon_id_col].map(area_map)
    intersection["weight"] = intersection["overlap_area"] / intersection["poly_area"]

    # Normalize per polygon so weights sum to 1.0
    weight_sum = intersection.groupby(polygon_id_col)["weight"].transform("sum")
    intersection["weight"] = intersection["weight"] / weight_sum

    return intersection[[polygon_id_col, "lat_center", "lon_center", "weight"]].reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _daily_cache_path(cache_dir: str, year: int, month: int, day: int) -> Path:
    return Path(cache_dir) / "nldas" / str(year) / f"{year}{month:02d}{day:02d}.nc"


def _parse_giovanni_response(text: str) -> pd.Series:
    """Parse a CSV response from the Giovanni Time Series Service API.

    The response has 13 header lines as key,value pairs followed by
    standard CSV data with Timestamp and value columns.

    Parameters
    ----------
    text:
        Raw CSV response text from the Giovanni API.

    Returns
    -------
    pandas.Series
        Series with DatetimeIndex and float values.
    """
    lines = text.splitlines()

    # Find the index where header lines end (first line starting with 'Timestamp')
    header_end = next(i for i, line in enumerate(lines) if line.startswith("Timestamp"))
    data_lines = lines[header_end:]

    # Parse the CSV data from the data lines
    csv_text = "\n".join(data_lines)
    df = pd.read_csv(
        io.StringIO(csv_text),
        header=0,
        parse_dates=[0],
    )

    # Return as Series with DatetimeIndex
    ts_col = df.columns[0]
    val_col = df.columns[1]
    return pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.DatetimeIndex(df[ts_col]),
        dtype=float,
    )


def _fetch_giovanni_cell(
    lat: float,
    lon: float,
    variable: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    _token: str | None = None,
) -> str:
    """Fetch raw text from the Giovanni Time Series API for a single cell.

    This function performs only the HTTP request and returns the raw
    response text.  It does **not** cache or parse the response.

    Parameters
    ----------
    lat, lon:
        Grid cell center coordinates (EPSG:4326).
    variable:
        Short variable name (e.g. ``"APCP"``).
    start_date:
        Start of the temporal range (e.g. ``"2010-01-01T00:00:00"``).
    end_date:
        End of the temporal range (e.g. ``"2010-12-31T23:00:00"``).
    max_retries:
        Number of retry attempts with exponential back-off on 429/5xx.
    _token:
        Pre-fetched Earthdata bearer token. If None, obtains one via
        ``earthaccess.login()`` and ``earthaccess.get_edl_token()``.

    Returns
    -------
    str
        Raw CSV response text from the Giovanni API.
    """
    import earthaccess
    import requests as req

    if variable not in _GIOVANNI_VARIABLES:
        raise ValueError(
            f"Unknown variable {variable!r}. "
            f"Available: {list(_GIOVANNI_VARIABLES)}"
        )

    # Get auth token
    if _token is None:
        earthaccess.login()
        _token = earthaccess.get_edl_token()["access_token"]

    params = {
        "data": _GIOVANNI_VARIABLES[variable],
        "location": f"[{lat},{lon}]",
        "time": f"{start_date}/{end_date}",
    }
    headers = {"Authorization": f"Bearer {_token}"}

    for attempt in range(max_retries):
        try:
            resp = req.get(
                _GIOVANNI_URL, params=params, headers=headers, timeout=60
            )
            resp.raise_for_status()
            text = resp.text
            break
        except req.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response else 0
            if code == 429 or code >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "Giovanni HTTP %d for %s (lat=%s lon=%s); retrying in %ds",
                    code, variable, lat, lon, wait,
                )
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise
        except req.exceptions.RequestException as exc:
            wait = 2 ** attempt
            logger.warning(
                "Giovanni request error for %s (lat=%s lon=%s): %s; retrying in %ds",
                variable, lat, lon, exc, wait,
            )
            time.sleep(wait)
            if attempt == max_retries - 1:
                raise

    return text


def _cache_giovanni_response(
    lat: float,
    lon: float,
    variable: str,
    start_date: str,
    end_date: str,
    cache_dir: str | None = None,
    max_retries: int = 3,
    _token: str | None = None,
) -> str:
    """Return raw Giovanni response text, using a file cache when available.

    If *cache_dir* is provided and a cached file exists for the requested
    cell/variable/date-range combination, the cached text is returned
    without making an HTTP request.  Otherwise :func:`_fetch_giovanni_cell`
    is called and the response is written to the cache before being
    returned.

    Parameters
    ----------
    lat, lon:
        Grid cell center coordinates (EPSG:4326).
    variable:
        Short variable name (e.g. ``"APCP"``).
    start_date, end_date:
        Temporal range strings (e.g. ``"2010-01-01T00:00:00"``).
    cache_dir:
        Optional directory for per-cell text cache files.
    max_retries:
        Passed through to :func:`_fetch_giovanni_cell`.
    _token:
        Pre-fetched Earthdata bearer token.

    Returns
    -------
    str
        Raw CSV response text (from cache or freshly fetched).
    """
    # Build a short cache key from the date range (strip time portion)
    start_tag = start_date[:10].replace("-", "")
    end_tag = end_date[:10].replace("-", "")
    cache_key = f"{lat}_{lon}_{variable}_{start_tag}_{end_tag}.txt"

    if cache_dir is not None:
        cache_path = Path(cache_dir) / "giovanni" / cache_key
        if cache_path.exists():
            return cache_path.read_text()

    text = _fetch_giovanni_cell(
        lat=lat, lon=lon, variable=variable,
        start_date=start_date, end_date=end_date,
        max_retries=max_retries, _token=_token,
    )

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)

    return text
