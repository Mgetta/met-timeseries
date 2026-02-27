"""Area-weighted zonal statistics using exactextract."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def aggregate_to_polygons(
    raster: xr.DataArray,
    polygons: gpd.GeoDataFrame,
    stat: str = "mean",
    id_column: str = "polygon_id",
) -> pd.DataFrame:
    """Compute area-weighted zonal statistics for each polygon.

    Wraps :func:`exactextract.exact_extract` to aggregate a gridded
    raster to the polygon geometries using precise area-weighted
    statistics.

    Parameters
    ----------
    raster : xr.DataArray
        2-D (lat x lon) or 3-D (time x lat x lon) raster DataArray.
        Must have CRS information accessible via ``rioxarray`` (i.e.,
        ``raster.rio.crs`` must be set).
    polygons : gpd.GeoDataFrame
        GeoDataFrame of polygon geometries.  The CRS will be aligned to
        the raster CRS automatically.
    stat : str
        Statistic to compute.  Supported values are any statistic
        accepted by ``exactextract`` (e.g. ``"mean"``, ``"sum"``,
        ``"min"``, ``"max"``).  Default is ``"mean"``.
    id_column : str
        Column in *polygons* that contains unique polygon identifiers.
        Default is ``"polygon_id"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per polygon and one column per time step
        (for 3-D rasters) or a single ``value`` column (for 2-D rasters).
        The polygon ID column is included as a column.

    Raises
    ------
    ImportError
        If the ``exactextract`` package is not installed.
    """
    try:
        from exactextract import exact_extract  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "exactextract is required for polygon aggregation. "
            "Install it with: pip install exactextract"
        ) from exc

    # Align CRS
    raster_crs = raster.rio.crs
    if raster_crs is not None and polygons.crs != raster_crs:
        polygons = polygons.to_crs(raster_crs)

    logger.debug("Running exact_extract with stat=%r on %d polygons", stat, len(polygons))

    results = exact_extract(raster, polygons, stat, include_cols=[id_column], output="pandas")
    return results
