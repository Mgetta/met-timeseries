"""
NLDAS-2 data source.

The public API is :func:`fetch_nldas`, which downloads NLDAS-2 hourly forcings
for a given bounding box, year, and month and returns an aggregated monthly
:class:`xarray.Dataset`.

Two fetch strategies are available:

* ``"grid"`` (default) — uses the ``earthaccess`` library to download full
  spatial grids.  Authentication is handled automatically via ``~/.netrc``,
  environment variables (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``), or
  an interactive prompt.  A free NASA Earthdata account is required:
  https://urs.earthdata.nasa.gov

* ``"datarods"`` — uses NASA's Hydrology Data Rods service to fetch
  time-series for individual grid cells specified by a pre-computed weight
  table.  This transfers only the data that overlaps the polygons of interest
  and does not require authentication.
"""

from __future__ import annotations

import calendar
import datetime as dt
import io
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from met_timeseries.sources.base import BoundingBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Rods constants
# ---------------------------------------------------------------------------

_DATARODS_URL = "https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi"

#: Mapping from short variable names to Data Rods variable identifiers
_DATARODS_VARIABLES: dict[str, str] = {
    "TMP": "NLDAS:NLDAS_FORA0125_H.002:TMP2m",
    "SPFH": "NLDAS:NLDAS_FORA0125_H.002:SPFH2m",
    "PRES": "NLDAS:NLDAS_FORA0125_H.002:PRESsfc",
    "UGRD": "NLDAS:NLDAS_FORA0125_H.002:UGRD10m",
    "VGRD": "NLDAS:NLDAS_FORA0125_H.002:VGRD10m",
    "DLWRF": "NLDAS:NLDAS_FORA0125_H.002:DLWRFsfc",
    "CAPE": "NLDAS:NLDAS_FORA0125_H.002:CAPEsfc",
    "PEVAP": "NLDAS:NLDAS_FORA0125_H.002:PEVAPsfc",
    "APCP": "NLDAS:NLDAS_FORA0125_H.002:APCPsfc",
    "DSWRF": "NLDAS:NLDAS_FORA0125_H.002:DSWRFsfc",
}


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


def fetch_nldas(
    bounds: BoundingBox,
    year: int,
    month: int,
    variables: list[str] | None = None,
    cache_dir: str | None = None,
    strategy: str = "grid",
    weights: pd.DataFrame | None = None,
    polygon_id_col: str = "metzone_id",
) -> xr.Dataset | dict[str, pd.DataFrame]:
    """Fetch NLDAS-2 hourly forcing data for the given bounding box.

    Two fetch strategies are supported:

    * ``"grid"`` (default) — downloads full spatial grids via ``earthaccess``.
      Authentication is handled automatically by :func:`earthaccess.login`,
      which reads ``~/.netrc``, environment variables
      (``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``), or prompts
      interactively.  A free NASA Earthdata account is required:
      https://urs.earthdata.nasa.gov

    * ``"datarods"`` — fetches per-cell hourly timeseries from NASA's
      Hydrology Data Rods service, then applies pre-computed area-overlap
      weights to produce polygon-level weighted averages.  No authentication
      is required.  The *weights* parameter must be provided.

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
        If provided, results are cached here and loaded on subsequent calls
        for the same month.
    strategy:
        Fetch strategy.  ``"grid"`` uses earthaccess (returns
        :class:`xarray.Dataset`); ``"datarods"`` uses the Data Rods API
        (returns ``dict[polygon_id, pd.DataFrame]``).
    weights:
        Pre-computed weight table from :func:`compute_nldas_weights`.
        Required when *strategy* is ``"datarods"``.
    polygon_id_col:
        Column name for polygon identifier in *weights*.  Used only when
        *strategy* is ``"datarods"``.

    Returns
    -------
    xarray.Dataset or dict[str, pandas.DataFrame]
        When *strategy* is ``"grid"``: hourly Dataset with ``time``, ``lat``,
        and ``lon`` dimensions.
        When *strategy* is ``"datarods"``: mapping of polygon_id →
        DataFrame with ``datetime`` index and one column per variable.

    Raises
    ------
    RuntimeError
        If no granules are found for the requested period (grid strategy).
    ValueError
        If *strategy* is ``"datarods"`` but *weights* is not provided, or
        if an unknown *strategy* is given.
    """
    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    if strategy == "datarods":
        if weights is None:
            raise ValueError("weights must be provided when strategy='datarods'")
        return fetch_nldas_datarods(
            weights=weights,
            year=year,
            month=month,
            variables=variables,
            polygon_id_col=polygon_id_col,
            cache_dir=cache_dir,
        )
    elif strategy != "grid":
        raise ValueError(f"Unknown strategy {strategy!r}. Must be 'grid' or 'datarods'.")

    import earthaccess

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
# Grid generation
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


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

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

    # Compute polygon areas (in projected units — use equal-area for accuracy)
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
# Data Rods fetch
# ---------------------------------------------------------------------------

def _parse_datarods_response(text: str) -> pd.Series:
    """Parse an ASCII ``asc2`` response from the Data Rods API.

    The format looks like::

        Date&Time       Data
        2010-01-01T00   0.0000
        2010-01-01T01   0.0000
        ...

    Parameters
    ----------
    text:
        Raw response text from the Data Rods API.

    Returns
    -------
    pandas.Series
        Series with :class:`pandas.DatetimeIndex` and float values.
    """
    df = pd.read_csv(
        io.StringIO(text),
        sep=r"\s+",
        comment="#",
        header=0,
    )
    # Normalise column names: strip whitespace
    df.columns = [c.strip() for c in df.columns]
    # Locate the timestamp column (first) and data column (second)
    ts_col = df.columns[0]
    data_col = df.columns[1]
    timestamps = pd.to_datetime(df[ts_col], format="%Y-%m-%dT%H")
    values = pd.to_numeric(df[data_col], errors="coerce")
    return pd.Series(values.values, index=timestamps, dtype=float)


def _fetch_datarods_cell(
    lat: float,
    lon: float,
    variable: str,
    year: int,
    month: int,
    cache_dir: str | None = None,
    max_retries: int = 3,
) -> pd.Series:
    """Fetch a single-cell timeseries from the Data Rods API.

    Parameters
    ----------
    lat, lon:
        Grid cell center coordinates.
    variable:
        Short variable name (e.g. ``"APCP"``).
    year, month:
        Temporal period.
    cache_dir:
        Optional cache directory.
    max_retries:
        Number of retry attempts with exponential back-off.

    Returns
    -------
    pandas.Series
        Hourly timeseries for the cell.
    """
    if variable not in _DATARODS_VARIABLES:
        raise ValueError(
            f"Unknown variable {variable!r}. Available: {list(_DATARODS_VARIABLES)}"
        )

    # Cache path
    if cache_dir is not None:
        cache_path = (
            Path(cache_dir)
            / "datarods"
            / f"{lat}_{lon}_{variable}_{year}{month:02d}.csv"
        )
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]

    _, n_days = calendar.monthrange(year, month)
    start_date = f"{year}-{month:02d}-01T00"
    end_date = f"{year}-{month:02d}-{n_days:02d}T23"

    params = urllib.parse.urlencode(
        {
            "variable": _DATARODS_VARIABLES[variable],
            "startDate": start_date,
            "endDate": end_date,
            "location": f"GEOM:POINT({lon},{lat})",
            "type": "asc2",
        }
    )
    url = f"{_DATARODS_URL}?{params}"

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                wait = 2 ** attempt
                logger.warning(
                    "Data Rods HTTP %d for %s (lat=%s lon=%s); retrying in %ds",
                    exc.code,
                    variable,
                    lat,
                    lon,
                    wait,
                )
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise
        except urllib.error.URLError as exc:
            wait = 2 ** attempt
            logger.warning(
                "Data Rods URL error for %s (lat=%s lon=%s): %s; retrying in %ds",
                variable,
                lat,
                lon,
                exc,
                wait,
            )
            time.sleep(wait)
            if attempt == max_retries - 1:
                raise

    series = _parse_datarods_response(text)

    if cache_dir is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(cache_path, header=True)

    return series


def fetch_nldas_datarods(
    weights: pd.DataFrame,
    year: int,
    month: int,
    variables: list[str] | None = None,
    polygon_id_col: str = "metzone_id",
    cache_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch NLDAS-2 hourly data using Data Rods for weighted grid cells.

    For each unique (lat, lon) in the weights table and each variable,
    fetches an hourly timeseries from the NASA Data Rods API. Returns
    one DataFrame per polygon with weighted-average timeseries.

    .. note::
        Nonlinear derivations (e.g. wind speed, cloud cover, PET) must be
        computed **per cell before spatial averaging** to preserve physical
        correctness.  Pass in pre-computed derived variables via *variables*
        if needed, or compute them on the returned per-polygon DataFrames.

    Parameters
    ----------
    weights:
        Pre-computed weight table from :func:`compute_nldas_weights`.
        Columns: ``[polygon_id_col, "lat_center", "lon_center", "weight"]``.
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    variables:
        NLDAS variable short names. Defaults to
        ``["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]``.
    polygon_id_col:
        Column name for polygon identifier in the weights table.
    cache_dir:
        If provided, per-cell timeseries are cached as CSV files.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of polygon_id → DataFrame with ``datetime`` index and
        one column per variable containing the weighted-average hourly
        timeseries.
    """
    if variables is None:
        variables = ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]

    unique_cells = weights[["lat_center", "lon_center"]].drop_duplicates()
    n_cells = len(unique_cells)
    logger.info(
        "Fetching Data Rods: %d unique cells × %d variables for %d-%02d",
        n_cells,
        len(variables),
        year,
        month,
    )

    # Fetch timeseries for every cell × variable combination
    cell_data: dict[tuple[float, float], dict[str, pd.Series]] = {}
    for i, (_, row) in enumerate(unique_cells.iterrows(), start=1):
        lat, lon = row["lat_center"], row["lon_center"]
        logger.debug("Fetching cell %d/%d: lat=%s lon=%s", i, n_cells, lat, lon)
        cell_data[(lat, lon)] = {}
        for var in variables:
            series = _fetch_datarods_cell(
                lat=lat,
                lon=lon,
                variable=var,
                year=year,
                month=month,
                cache_dir=cache_dir,
            )
            cell_data[(lat, lon)][var] = series

    # Compute weighted averages per polygon
    results: dict[str, pd.DataFrame] = {}
    for poly_id, group in weights.groupby(polygon_id_col):
        poly_frames: dict[str, pd.Series] = {}
        for var in variables:
            weighted_sum: pd.Series | None = None
            weight_total = 0.0
            for _, wrow in group.iterrows():
                lat, lon, w = wrow["lat_center"], wrow["lon_center"], wrow["weight"]
                series = cell_data.get((lat, lon), {}).get(var)
                if series is None:
                    continue
                if weighted_sum is None:
                    weighted_sum = series * w
                else:
                    weighted_sum = weighted_sum + series * w
                weight_total += w
            if weighted_sum is not None and weight_total > 0:
                # Divide by weight_total (normally 1.0 after normalization)
                # to handle partial data where some cells return no data.
                poly_frames[var] = weighted_sum / weight_total
        if poly_frames:
            results[str(poly_id)] = pd.DataFrame(poly_frames)
        else:
            results[str(poly_id)] = pd.DataFrame()

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _monthly_cache_path(cache_dir: str, year: int, month: int) -> Path:
    return Path(cache_dir) / "nldas" / f"{year}{month:02d}.nc"
