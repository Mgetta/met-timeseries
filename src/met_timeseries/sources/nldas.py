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


def download_datarods(
    bounds: BoundingBox,
    year: int,
    month: int,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
) -> dict[tuple[float, float], dict[str, pd.Series]]:
    """Download raw NLDAS-2 per-cell timeseries via Giovanni Time Series API.

    This function is independent of any weighting.  It determines which
    NLDAS grid cells fall within *bounds*, authenticates with NASA Earthdata
    once, then fetches hourly timeseries for each cell.  Raw per-cell data
    is optionally cached as CSV files.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326 used to select grid cells.
    year:
        Calendar year (e.g. 2010).
    month:
        Calendar month (1–12).
    variables:
        NLDAS-2 short variable names to fetch. Defaults to
        ``["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    cache_dir:
        Optional directory for per-cell CSV cache files.

    Returns
    -------
    dict[tuple[float, float], dict[str, pandas.Series]]
        Mapping from ``(lat, lon)`` to a dict of variable name →
        hourly ``pd.Series`` with ``DatetimeIndex``.
    """
    import earthaccess

    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    # Determine NLDAS grid cells that fall within the bounding box
    buf = 0.125  # one cell buffer
    grid = generate_nldas_grid(
        west=bounds.west - buf,
        south=bounds.south - buf,
        east=bounds.east + buf,
        north=bounds.north + buf,
    )
    bbox_polygon = box(bounds.west, bounds.south, bounds.east, bounds.north)
    grid_in_bounds = grid[grid.intersects(bbox_polygon)]
    unique_cells = grid_in_bounds[["lat_center", "lon_center"]].drop_duplicates()

    # Authenticate once and reuse the token for all requests
    earthaccess.login()
    token = earthaccess.get_edl_token()["access_token"]

    n_cells = len(unique_cells)
    logger.info(
        "Downloading Giovanni timeseries: %d unique cells × %d variables for %d-%02d",
        n_cells, len(variables), year, month,
    )

    cell_data: dict[tuple[float, float], dict[str, pd.Series]] = {}
    for i, (_, row) in enumerate(unique_cells.iterrows(), start=1):
        lat, lon = row["lat_center"], row["lon_center"]
        logger.debug("Fetching cell %d/%d: lat=%s lon=%s", i, n_cells, lat, lon)
        cell_data[(lat, lon)] = {}
        for var in variables:
            series = _fetch_giovanni_cell(
                lat=lat, lon=lon, variable=var,
                year=year, month=month,
                cache_dir=cache_dir, _token=token,
            )
            cell_data[(lat, lon)][var] = series

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
    year: int,
    month: int,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
    weights: pd.DataFrame | None = None,
    polygon_id_col: str = "metzone_id",
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
        Calendar year (e.g. 2010).
    month:
        Calendar month (1–12).
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
    )

    if weights is None:
        bbox_polygon = box(bounds.west, bounds.south, bounds.east, bounds.north)
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


def fetch_nldas_grid(
    bounds: BoundingBox,
    year: int,
    month: int,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
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
        Calendar year (e.g. 2010).
    month:
        Calendar month (1–12).
    variables:
        NLDAS-2 short variable names to include.  Defaults to ``["APCP",
        "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    cache_dir:
        If provided, a subsetted monthly NetCDF file is cached here and
        loaded on subsequent calls for the same month.

    Returns
    -------
    xarray.Dataset
        Hourly Dataset with ``time``, ``lat``, and ``lon`` dimensions.
        The ``time`` coordinate carries proper ``datetime64`` timestamps.

    Raises
    ------
    RuntimeError
        If no granules are found for the requested period.
    """
    import earthaccess

    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    logger.info("Fetching NLDAS-2 data: year=%d month=%d bounds=%r", year, month, bounds)

    # Check cache first
    if cache_dir is not None:
        cache_path = _monthly_cache_path(cache_dir, year, month)
        if cache_path.exists():
            logger.debug("Loading NLDAS-2 from cache: %s", cache_path)
            return xr.open_dataset(cache_path)

    # Authenticate with NASA Earthdata
    earthaccess.login()

    # Temporal range: full month
    _, n_days = calendar.monthrange(year, month)
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{n_days:02d}"

    # Search for NLDAS-2 granules
    results = earthaccess.search_data(
        short_name="NLDAS_FORA0125_H",
        version="2.0",
        temporal=(start, end),
        bounding_box=(bounds.west, bounds.south, bounds.east, bounds.north),
    )

    if not results:
        raise RuntimeError(f"No NLDAS-2 granules found for {year}-{month:02d}")

    logger.debug("Found %d NLDAS-2 granules", len(results))

    # Open all granules as fsspec file objects
    file_objs = earthaccess.open(results)

    # Load all granules into a single Dataset
    ds = xr.open_mfdataset(
        file_objs,
        engine="h5netcdf",
        combine="nested",
        concat_dim="time",
    )

    # Subset variables
    ds = ds[variables]

    # Subset spatial domain
    ds = ds.sel(
        lat=slice(bounds.south, bounds.north),
        lon=slice(bounds.west, bounds.east),
    )

    # Ensure time coordinate has proper datetime64 values
    if "time" not in ds.coords or not np.issubdtype(ds["time"].dtype, np.datetime64):
        n_hours = ds.sizes["time"]
        start_ts = np.datetime64(dt.datetime(year, month, 1), "ns")
        timestamps = [start_ts + np.timedelta64(h, "h") for h in range(n_hours)]
        ds = ds.assign_coords(time=timestamps)

    # Save to cache
    if cache_dir is not None:
        cache_path = _monthly_cache_path(cache_dir, year, month)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache_path)
        logger.debug("Cached NLDAS-2 data to: %s", cache_path)

    return ds


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

def _monthly_cache_path(cache_dir: str, year: int, month: int) -> Path:
    return Path(cache_dir) / "nldas" / f"{year}{month:02d}.nc"


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
    with io.StringIO(text) as f:
        # First _GIOVANNI_HEADER_LINES lines are metadata key,value pairs; skip them
        for _ in range(_GIOVANNI_HEADER_LINES):
            f.readline()

        # Read the CSV data (has a header row with column names)
        df = pd.read_csv(
            f,
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
    year: int,
    month: int,
    cache_dir: str | None = None,
    max_retries: int = 3,
    _token: str | None = None,
) -> pd.Series:
    """Fetch a single-cell timeseries from the Giovanni Time Series API.

    Parameters
    ----------
    lat, lon:
        Grid cell center coordinates (EPSG:4326).
    variable:
        Short variable name (e.g. ``"APCP"``).
    year, month:
        Temporal period.
    cache_dir:
        Optional cache directory for per-cell CSV files.
    max_retries:
        Number of retry attempts with exponential back-off on 429/5xx.
    _token:
        Pre-fetched Earthdata bearer token. If None, obtains one via
        ``earthaccess.login()`` and ``earthaccess.get_edl_token()``.

    Returns
    -------
    pandas.Series
        Hourly timeseries for the cell with DatetimeIndex.
    """
    import earthaccess
    import requests as req

    if variable not in _GIOVANNI_VARIABLES:
        raise ValueError(
            f"Unknown variable {variable!r}. "
            f"Available: {list(_GIOVANNI_VARIABLES)}"
        )

    # Check cache
    if cache_dir is not None:
        cache_path = (
            Path(cache_dir) / "giovanni"
            / f"{lat}_{lon}_{variable}_{year}{month:02d}.csv"
        )
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]

    # Get auth token
    if _token is None:
        earthaccess.login()
        _token = earthaccess.get_edl_token()["access_token"]

    _, n_days = calendar.monthrange(year, month)
    time_start = f"{year}-{month:02d}-01T00:00:00"
    time_end = f"{year}-{month:02d}-{n_days:02d}T23:00:00"

    params = {
        "data": _GIOVANNI_VARIABLES[variable],
        "location": f"[{lat},{lon}]",
        "time": f"{time_start}/{time_end}",
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

    series = _parse_giovanni_response(text)

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(cache_path, header=True)

    return series
