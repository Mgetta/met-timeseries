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

_CACHE_PREFIX = "NLDAS_FORA0125_H_"

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
    
_GRID_VARIABLES: dict[str, str] = {
    "TMP": "Tair",
    "SPFH": "Qair",
    "PRES": "PSurf",
    "UGRD": "Wind_E",
    "VGRD": "Wind_N",
    "DLWRF": "LWdown",
    "PEVAP": "PotEvap",
    "APCP": "Rainf",
    "DSWRF": "SWdown",
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



def fetch_nldas(
    bounds: BoundingBox,
    start: str,
    cache_dir: str,
    end: str | None = None,
    variables: list[str] | None = None,
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
    import earthaccess

    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end) if end is not None else start_date
    if end_date < start_date:
        raise ValueError(
            f"end ({end!r}) must not be before start ({start!r})"
        )

    days = pd.date_range(start=start_date, end=end_date, freq="D")

    earthaccess.login()

    for date in days:
        cache_path = Path(cache_dir) / f"{_CACHE_PREFIX}{date.strftime('%Y%m%d')}.nc"
        if cache_path.exists():
            ds = xr.open_dataset(cache_path)
        else:
            ds = _fetch_nldas_granules(
                bounds, str(date.date()), str(date.date()), variables, max_connections
            )
            _cache_dataset(ds, cache_path)

    return ds


def _open_file_obj(file_obj: object) -> xr.Dataset:
    """Open a file-like object as an xarray Dataset. Note that this is an in-memory operation and does not write to disk."""
    return xr.open_dataset(file_obj, engine="h5netcdf")


def _clip_dataset(
    ds: xr.Dataset,
    bounds: BoundingBox,
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
    bounds: BoundingBox,
    start_date: str,
    end_date: str,
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

    results = earthaccess.search_data(
        short_name="NLDAS_FORA0125_H",
        version="2.0",
        temporal=(start_date, end_date),
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


def _fetch_nldas_granules(
    bounds: BoundingBox,
    start_date: str,
    end_date: str,
    variables: list[str],
    max_connections: int,
    cache_dir: Path = None
) -> xr.Dataset:
    """Download and subset all granules for a date range, returning a Dataset.

    Opens granules concurrently using :class:`concurrent.futures.ThreadPoolExecutor`.
    Failed granules are skipped with a warning; a :exc:`RuntimeError` is raised
    only when *all* granules for the range fail.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
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

    results = _search_nldas_granules(bounds, start_date, end_date)

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

    # Some NLDAS-2 granules store time as a numeric offset rather than
    # datetime64. Reconstruct the coordinate from the known range start if
    # the decoded dtype is not already datetime64.
    if "time" not in ds.coords or not np.issubdtype(ds["time"].dtype, np.datetime64):
        n_hours = ds.sizes["time"]
        start_dt = dt.datetime.fromisoformat(start_date)
        start_ts = np.datetime64(start_dt, "ns")
        timestamps = [start_ts + np.timedelta64(h, "h") for h in range(n_hours)]
        ds = ds.assign_coords(time=timestamps)

    return ds

def _cache_dataset(ds: xr.Dataset,cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for var in ds.data_vars
    }
    ds.to_netcdf(cache_path, encoding=encoding)
    logger.debug("Cached NLDAS data to %s", cache_path)
    return cache_path


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

# Datarods-based NLDAS processing

def process_nldas(
    bounds: BoundingBox,
    start: str = "1995-01-01",
    end: str | None = None,
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
    start:
        Start date in ``"YYYY-MM-DD"`` format.  Defaults to ``"1995-01-01"``.
    end:
        End date in ``"YYYY-MM-DD"`` format (inclusive).  When ``None``
        (the default), only the single day given by *start* is fetched.
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
        start=start,
        end=end,
        variables=variables,
        cache_dir=cache_dir,
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



def download_datarods(
    bounds: BoundingBox,
    start: str = "1995-01-01",
    end: str | None = None,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
) -> dict[tuple[float, float], dict[str, pd.Series]]:
    """Download raw NLDAS-2 per-cell timeseries via Giovanni Time Series API.

    This function is independent of any weighting.  It determines which
    NLDAS grid cells fall within *bounds*, authenticates with NASA Earthdata
    once, then fetches hourly timeseries for each cell.  Raw per-cell data
    is optionally cached as CSV files.

    Data is requested in yearly chunks (one API call per year) to minimise
    the total number of HTTP requests.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326 used to select grid cells.
    start:
        Start date in ``"YYYY-MM-DD"`` format.  Defaults to ``"1995-01-01"``.
    end:
        End date in ``"YYYY-MM-DD"`` format (inclusive).  When ``None``
        (the default), only the single day given by *start* is fetched.
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

    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end) if end is not None else start_date

    if end_date < start_date:
        raise ValueError(
            f"end ({end!r}) must not be before start ({start!r})"
        )

    # Build list of (start_date, end_date) pairs in yearly chunks to minimise
    # the total number of API calls.
    date_ranges: list[tuple[str, str]] = []
    for y in range(start_date.year, end_date.year + 1):
        chunk_start = start_date if y == start_date.year else dt.date(y, 1, 1)
        chunk_end = end_date if y == end_date.year else dt.date(y, 12, 31)
        date_ranges.append(
            (f"{chunk_start.isoformat()}T00:00:00",
             f"{chunk_end.isoformat()}T23:00:00"),
        )

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



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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



