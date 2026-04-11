"""
Zonal statistics: aggregate raster data over a polygon.
"""
"""
Zonal statistics: aggregate gridded data over polygons.

The core weight computation (:func:`compute_weights`) is source-agnostic —
it works with any regular lat/lon grid.  Higher-level helpers build on it:

* :func:`aggregate_over_polygon` — weighted mean of an xarray Dataset
  (replaces the previous implementation)
* :func:`weighted_mean_timeseries` — collapse (time, lat, lon) → (time,)
  returning a pandas Series per variable, convenient for disaggregation
"""


import functools
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pyproj import Transformer
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Equal-area reprojection (module-level, created once on import)
# ---------------------------------------------------------------------------

_TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def _to_ea(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a geometry from EPSG:4326 to EPSG:6933 (equal-area)."""
    return shapely_transform(_TO_EQUAL_AREA.transform, geom)


# ---------------------------------------------------------------------------
# Core weight computation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _compute_weights_cached(
    polygon_wkb: bytes,
    lats: tuple[float, ...],
    lons: tuple[float, ...],
    normalize: bool,
) -> np.ndarray:
    """LRU-cached implementation; keyed on hashable WKB + coordinate tuples."""
    from shapely import from_wkb

    polygon: BaseGeometry = from_wkb(polygon_wkb)

    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)

    if len(lats_arr) < 2:
        raise ValueError(
            "Cannot infer cell height from a single latitude value; "
            "at least 2 latitude coordinates are required."
        )
    if len(lons_arr) < 2:
        raise ValueError(
            "Cannot infer cell width from a single longitude value; "
            "at least 2 longitude coordinates are required."
        )

    dx = abs(float(lons_arr[1] - lons_arr[0]))
    dy = abs(float(lats_arr[1] - lats_arr[0]))

    weights = np.zeros((len(lats_arr), len(lons_arr)), dtype=float)

    for i, lat in enumerate(lats_arr):
        for j, lon in enumerate(lons_arr):
            cell = shapely_box(
                lon - dx / 2, lat - dy / 2, lon + dx / 2, lat + dy / 2
            )
            intersection = cell.intersection(polygon)
            if intersection.is_empty:
                continue

            cell_ea = _to_ea(cell)
            inter_ea = _to_ea(intersection)
            weights[i, j] = inter_ea.area / cell_ea.area

    if normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total

    return weights


def compute_weights(
    polygon: BaseGeometry,
    lats: tuple[float, ...],
    lons: tuple[float, ...],
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Return a 2-D weight array (shape ``len(lats) × len(lons)``).

    Each weight represents the fractional area overlap between a grid cell
    and the polygon, computed in an equal-area projection (EPSG:6933).

    Results are LRU-cached on the polygon's WKB representation and the
    coordinate tuples, so repeated calls with the same inputs are free.

    Parameters
    ----------
    polygon:
        Shapely geometry in EPSG:4326.
    lats:
        Grid cell center latitudes (tuple of floats).
    lons:
        Grid cell center longitudes (tuple of floats).
    normalize:
        If ``False`` (default), each weight is ``intersection_area /
        cell_area`` — the fraction of the *cell* covered by the polygon.
        The caller must divide by ``weights.sum()`` at application time.

        If ``True``, weights are rescaled so they sum to 1.0.  The result
        can be used directly as ``(data * weights).sum()`` without further
        normalization.

    Returns
    -------
    numpy.ndarray
        2-D float array aligned to ``(lat, lon)``.

    Raises
    ------
    ValueError
        If *lats* or *lons* contain fewer than 2 values (cell spacing
        cannot be inferred from a single coordinate).
    """
    return _compute_weights_cached(polygon.wkb, lats, lons, normalize)



# ---------------------------------------------------------------------------
# High-level aggregation functions
# ---------------------------------------------------------------------------

def plot_weights(
    polygon: BaseGeometry,
    dataset: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    normalize: bool = True,
) -> "matplotlib.axes.Axes":
    """Plot the weight grid for a polygon overlaid on the dataset grid."""
    import matplotlib.pyplot as plt

    w_ds = get_weights(polygon, dataset, lat_dim=lat_dim, lon_dim=lon_dim, normalize=normalize)

    weights = w_ds["weight"].where(w_ds["weight"] > 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("lightgrey")

    weights.plot.pcolormesh(
        x=lon_dim, y=lat_dim,
        ax=ax,
        cmap="YlOrRd",
        edgecolors="grey",
        linewidths=0.3,
        cbar_kwargs={"label": "Weight", "shrink": 0.8},
    )

    gpd.GeoSeries([polygon], crs="EPSG:4326").boundary.plot(
        ax=ax, color="blue", linewidth=2, label="Polygon of interest",
    )

    # extent = union of grid bounds + polygon bounds, with padding
    lats = dataset[lat_dim].values
    lons = dataset[lon_dim].values
    dy = abs(float(lats[1] - lats[0])) / 2
    dx = abs(float(lons[1] - lons[0])) / 2
    poly_bounds = polygon.bounds  # (minx, miny, maxx, maxy)

    xmin = min(lons.min() - dx, poly_bounds[0]) - dx
    xmax = max(lons.max() + dx, poly_bounds[2]) + dx
    ymin = min(lats.min() - dy, poly_bounds[1]) - dy
    ymax = max(lats.max() + dy, poly_bounds[3]) + dy

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.set_title("Grid-Cell Weights")

    return ax

def get_weights(
    polygon: BaseGeometry,
    dataset: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    normalize: bool = True,
) -> xr.Dataset:
    """Return grid-cell weights as an xarray Dataset aligned to the input grid.

    The returned Dataset has a single variable ``weight`` with the same
    lat/lon coordinates as *dataset*.
    """
    lats = dataset[lat_dim].values
    lons = dataset[lon_dim].values

    w = compute_weights(
        polygon,
        tuple(float(v) for v in lats),
        tuple(float(v) for v in lons),
        normalize=normalize,
    )

    return xr.Dataset(
        {"weight": xr.DataArray(w, dims=[lat_dim, lon_dim], coords={lat_dim: lats, lon_dim: lons})},
    )

def get_greid(
    dataset: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of grid-cell polygons from an xarray Dataset.

    Columns: lat, lon, row_id, column_id, geometry, test
    """
    lats = dataset[lat_dim].values
    lons = dataset[lon_dim].values

    dy = abs(float(lats[1] - lats[0]))
    dx = abs(float(lons[1] - lons[0]))
    half_x = dx / 2.0
    half_y = dy / 2.0

    records = []
    for row_id, lat in enumerate(lats):
        for col_id, lon in enumerate(lons):
            records.append({
                "lat": float(lat),
                "lon": float(lon),
                "row_id": row_id,
                "column_id": col_id,
                "geometry": shapely_box(
                    lon - half_x, lat - half_y,
                    lon + half_x, lat + half_y,
                ),
            })

    return gpd.GeoDataFrame(records, crs="EPSG:4326")

 
def aggregate_over_polygon(
    dataset: xr.Dataset,
    polygon: BaseGeometry,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> dict[str, list[float] | float]:
    """Compute the spatial weighted mean of each variable over *polygon*.

    Each grid cell's contribution is proportional to the fraction of its
    area that overlaps *polygon*.  If no cells overlap the polygon the
    function falls back to the nearest grid point to the polygon centroid.

    Parameters
    ----------
    dataset:
        xarray Dataset with at least *lat_dim* and *lon_dim* coordinates.
    polygon:
        Shapely geometry for the area of interest.
    lat_dim, lon_dim:
        Dimension names.

    Returns
    -------
    dict
        Variable name → scalar (2-D) or list of floats (3-D with time).
    """
    lats = dataset[lat_dim].values
    lons = dataset[lon_dim].values

    w = compute_weights(
        polygon,
        tuple(float(v) for v in lats),
        tuple(float(v) for v in lons),
        normalize=False,
    )

    total_weight = w.sum()

    if total_weight == 0:
        cx, cy = polygon.centroid.x, polygon.centroid.y
        lat_idx = int(np.argmin(np.abs(lats - cy)))
        lon_idx = int(np.argmin(np.abs(lons - cx)))
        w = np.zeros_like(w)
        w[lat_idx, lon_idx] = 1.0
        total_weight = 1.0

    result: dict[str, list[float] | float] = {}
    for var in dataset.data_vars:
        arr = dataset[var].values
        if arr.ndim == 2:
            result[str(var)] = float(np.nansum(arr * w) / total_weight)
        elif arr.ndim == 3:
            result[str(var)] = (
                np.nansum(arr * w[np.newaxis, :, :], axis=(1, 2)) / total_weight
            ).tolist()
        else:
            values = arr.ravel()
            result[str(var)] = (
                float(np.nanmean(values)) if len(values) > 0 else float("nan")
            )

    return result


def weighted_mean_timeseries(
    dataset: xr.Dataset,
    polygon: BaseGeometry,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    time_dim: str = "time",
) -> dict[str, pd.Series]:
    """Collapse a (time, lat, lon) Dataset to one Series per variable.

    This is the primary entry point for producing polygon-level timeseries
    that can be fed into the disaggregation pipeline.

    Parameters
    ----------
    dataset:
        xarray Dataset with ``(time, lat, lon)`` dimensions.
    polygon:
        Shapely geometry for the area of interest.
    lat_dim, lon_dim, time_dim:
        Dimension names.

    Returns
    -------
    dict[str, pandas.Series]
        Variable name → Series with DatetimeIndex.
    """
    lats = dataset[lat_dim].values
    lons = dataset[lon_dim].values

    w = compute_weights(
        polygon,
        tuple(float(v) for v in lats),
        tuple(float(v) for v in lons),
        normalize=True,
    )

    total = w.sum()
    if total == 0:
        cx, cy = polygon.centroid.x, polygon.centroid.y
        lat_idx = int(np.argmin(np.abs(lats - cy)))
        lon_idx = int(np.argmin(np.abs(lons - cx)))
        w = np.zeros_like(w)
        w[lat_idx, lon_idx] = 1.0

    times = pd.DatetimeIndex(dataset[time_dim].values)

    result: dict[str, pd.Series] = {}
    for var in dataset.data_vars:
        arr = dataset[var].values
        if arr.ndim == 3:
            values = np.nansum(arr * w[np.newaxis, :, :], axis=(1, 2))
        elif arr.ndim == 2:
            values = np.full(len(times), np.nansum(arr * w))
        else:
            values = np.full(len(times), float("nan"))
        result[str(var)] = pd.Series(values, index=times, name=var)

    return result





    
def _compute_weights_gpd(
    polygons: gpd.GeoDataFrame,
    polygon_id_col: str = "metzone_id",
    grid: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Compute area-overlap weights between polygons and the grid.

    For each polygon, finds all grid cells that intersect it and
    computes the fractional overlap area. Weights are normalized so they
    sum to 1.0 per polygon.

    Parameters
    ----------
    polygons:
        GeoDataFrame of user polygons (e.g., dissolved metzones).
    polygon_id_col:
        Column name for the polygon identifier.
    grid: gpd.GeoDataFrame | None = None,
        Pre-generated grid GeoDataFrame. If None, generates one
        clipped to the total bounds of *polygons* (with a small buffer).

    Returns
    -------
    pandas.DataFrame
        Columns: ``[polygon_id_col, "lat_center", "lon_center", "weight"]``.
        Weights sum to 1.0 per polygon.
    """

    # Ensure matching CRS
    polygons = polygons.to_crs("EPSG:4326")
    grid = grid.to_crs("EPSG:4326")

    # Compute polygon areas in equal-area projection for accuracy
    poly_areas = polygons.to_crs("EPSG:6933").geometry.area

    # Intersection
    intersection = gpd.overlay(
        polygons[[polygon_id_col, "geometry"]],
        grid[["lat_center", "lon_center", "geometry"]],
        how="intersection",
    )

    # Compute overlap area in equal-area projection
    intersection["overlap_area"] = intersection.to_crs("EPSG:6933").geometry.area


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
