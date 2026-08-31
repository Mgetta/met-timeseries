"""
Polygon loading and metzone dissolve logic.

The primary entry point is :func:`load_polygons`, which reads a vector file
containing individual model catchments, dissolves them by a metzone grouping
column, validates the geometries, and reprojects to the target CRS.
"""

from __future__ import annotations

import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


def load_polygons(
    catchment_path: str,
    metzone_column: str,
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Load a catchment vector file and dissolve catchments into metzones.

    Reads *catchment_path* with :func:`geopandas.read_file` (supports
    ``.shp``, ``.gpkg``, ``.geojson``, ``.parquet``, and any other format
    supported by GeoPandas / GDAL), then dissolves all catchments that share
    the same value in *metzone_column* into a single polygon.

    Parameters
    ----------
    catchment_path:
        Path to the vector file containing model catchments.
    metzone_column:
        Name of the attribute column used to group catchments into metzones.
    target_crs:
        CRS to reproject the dissolved metzone polygons to before returning.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per metzone with columns ``[metzone_column, "geometry"]``,
        projected to *target_crs*.

    Raises
    ------
    ValueError
        If *metzone_column* is not present in the vector file.
    """
    logger.info("Reading catchments from '%s'", catchment_path)
    if str(catchment_path).endswith(".parquet"):
        gdf = gpd.read_parquet(catchment_path)
    else:
        gdf = gpd.read_file(catchment_path)

    if metzone_column not in gdf.columns:
        available = ", ".join(gdf.columns.tolist())
        raise ValueError(
            f"Column '{metzone_column}' not found in '{catchment_path}'. "
            f"Available columns: {available}"
        )

    n_catchments = len(gdf)
    dissolved = gdf.dissolve(by=metzone_column).reset_index()
    n_metzones = len(dissolved)

    logger.info(
        "Dissolved %d catchments into %d metzones using column '%s'",
        n_catchments,
        n_metzones,
        metzone_column,
    )

    # Validate and fix geometries
    invalid_mask = ~dissolved.geometry.is_valid
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        logger.warning("Fixing %d invalid geometries with .buffer(0)", n_invalid)
        dissolved.loc[invalid_mask, "geometry"] = (
            dissolved.loc[invalid_mask, "geometry"].buffer(0)
        )

    # Reproject to target CRS
    if dissolved.crs is None:
        logger.warning("Input has no CRS; assuming %s", target_crs)
        dissolved = dissolved.set_crs(target_crs)
    else:
        dissolved = dissolved.to_crs(target_crs)

    return dissolved[[metzone_column, "geometry"]].copy()
