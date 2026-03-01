"""
Zonal statistics: aggregate raster data over a polygon.
"""

from __future__ import annotations

import functools

import numpy as np
import xarray as xr
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry


@functools.lru_cache(maxsize=128)
def _compute_weights(
    polygon_wkb: bytes,
    lats: tuple[float, ...],
    lons: tuple[float, ...],
) -> np.ndarray:
    """Return a 2-D weight array (shape ``len(lats) × len(lons)``).

    Each weight is the fraction of the corresponding grid-cell area that
    overlaps *polygon* (0.0 – 1.0).  Results are cached so that repeated
    calls for the same polygon / grid combination are free.
    """
    from shapely import from_wkb

    polygon: BaseGeometry = from_wkb(polygon_wkb)

    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)

    dx = abs(float(lons_arr[1] - lons_arr[0])) if len(lons_arr) > 1 else 0.125
    dy = abs(float(lats_arr[1] - lats_arr[0])) if len(lats_arr) > 1 else 0.125

    weights = np.zeros((len(lats_arr), len(lons_arr)), dtype=float)
    for i, lat in enumerate(lats_arr):
        for j, lon in enumerate(lons_arr):
            # shapely_box expects (minx, miny, maxx, maxy); lon maps to x, lat to y.
            cell = shapely_box(lon - dx / 2, lat - dy / 2, lon + dx / 2, lat + dy / 2)
            intersection_area = cell.intersection(polygon).area
            if intersection_area > 0:
                weights[i, j] = intersection_area / cell.area

    return weights


def aggregate_over_polygon(
    dataset: xr.Dataset,
    polygon: BaseGeometry,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> dict[str, float]:
    """Compute the area-weighted mean of each variable in *dataset* over *polygon*.

    Each grid cell's contribution is proportional to the fraction of its area
    that overlaps *polygon*.  If no cells overlap the polygon the function
    falls back to the nearest grid point to the polygon centroid.

    Parameters
    ----------
    dataset:
        An :class:`xarray.Dataset` with at least *lat_dim* and *lon_dim*
        coordinate arrays.
    polygon:
        A Shapely geometry representing the area of interest.
    lat_dim:
        Name of the latitude dimension in *dataset*.
    lon_dim:
        Name of the longitude dimension in *dataset*.

    Returns
    -------
    dict mapping variable name to its area-weighted mean scalar value over the
    polygon.
    """
    lats = dataset[lat_dim].values
    lons = dataset[lon_dim].values

    weights = _compute_weights(
        polygon.wkb,
        tuple(float(v) for v in lats),
        tuple(float(v) for v in lons),
    )

    total_weight = weights.sum()

    if total_weight == 0:
        # Fallback: nearest grid point to polygon centroid
        cx, cy = polygon.centroid.x, polygon.centroid.y
        lat_idx = int(np.argmin(np.abs(lats - cy)))
        lon_idx = int(np.argmin(np.abs(lons - cx)))
        weights = np.zeros_like(weights)
        weights[lat_idx, lon_idx] = 1.0
        total_weight = 1.0

    result: dict[str, float] = {}
    for var in dataset.data_vars:
        arr = dataset[var].values
        if arr.ndim == 2:
            spatial = arr
        elif arr.ndim == 3:
            # (time, lat, lon) — average over time first
            spatial = arr.mean(axis=0)
        else:
            spatial = arr.reshape(len(lats), len(lons))

        weighted_sum = float(np.nansum(spatial * weights))
        # Account for NaN cells by subtracting their weights from total
        nan_weight = float(np.nansum(weights[np.isnan(spatial)]))
        effective_weight = total_weight - nan_weight
        result[str(var)] = weighted_sum / effective_weight if effective_weight > 0 else float("nan")

    return result
