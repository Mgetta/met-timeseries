"""
Zonal statistics: aggregate raster data over a polygon.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry


def aggregate_over_polygon(
    dataset: xr.Dataset,
    polygon: BaseGeometry,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> dict[str, float]:
    """Compute the mean of each variable in *dataset* over *polygon*.

    Grid cells whose centres fall within *polygon* are selected and averaged.
    If no grid cells fall within the polygon the function falls back to the
    nearest grid point.

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
    dict mapping variable name to its mean scalar value over the polygon.
    """
    lats = dataset[lat_dim].values
    lons = dataset[lon_dim].values

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Build a boolean mask for cells inside the polygon
    try:
        from shapely import contains_xy

        mask = contains_xy(polygon, lon_grid.ravel(), lat_grid.ravel()).reshape(lon_grid.shape)
    except ImportError:
        from shapely.geometry import Point

        mask = np.array(
            [
                [polygon.contains(Point(lon_grid[i, j], lat_grid[i, j])) for j in range(lon_grid.shape[1])]
                for i in range(lon_grid.shape[0])
            ]
        )

    if not mask.any():
        # Fallback: nearest grid point to polygon centroid
        cx, cy = polygon.centroid.x, polygon.centroid.y
        lat_idx = int(np.argmin(np.abs(lats - cy)))
        lon_idx = int(np.argmin(np.abs(lons - cx)))
        mask[lat_idx, lon_idx] = True

    result: dict[str, float] = {}
    for var in dataset.data_vars:
        arr = dataset[var].values
        if arr.ndim == 2:
            values = arr[mask]
        elif arr.ndim == 3:
            # (time, lat, lon) or similar — take time-mean first, then spatial
            values = arr.mean(axis=0)[mask]
        else:
            values = arr.ravel()
        result[str(var)] = float(np.nanmean(values)) if len(values) > 0 else float("nan")

    return result
