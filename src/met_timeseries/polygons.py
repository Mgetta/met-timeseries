"""Polygon loading, validation, and CRS handling."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def load_polygons(
    path: Path | str,
    id_column: str = "polygon_id",
    target_crs: int = 4326,
) -> gpd.GeoDataFrame:
    """Load and validate a polygon GeoDataFrame from disk.

    Reads a GeoJSON, GeoPackage, or any fiona-supported vector format,
    re-projects to *target_crs* if necessary, and checks for required
    data quality constraints.

    Parameters
    ----------
    path : Path or str
        Path to the vector file (GeoJSON, GPKG, Shapefile, ...).
    id_column : str
        Column that contains unique polygon identifiers.
        Default is ``"polygon_id"``.
    target_crs : int
        EPSG code for the output CRS. Default is ``4326`` (WGS-84).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the geometry in *target_crs* and the
        ``polygon_id`` column guaranteed to contain unique, non-null
        values.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *id_column* is missing, contains nulls, or contains duplicate
        values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Polygon file not found: {path}")

    gdf = gpd.read_file(path)

    if id_column not in gdf.columns:
        raise ValueError(
            f"ID column {id_column!r} not found. "
            f"Available columns: {list(gdf.columns)}"
        )

    if gdf[id_column].isna().any():
        raise ValueError(f"ID column {id_column!r} contains null values.")

    if gdf[id_column].duplicated().any():
        dupes = gdf.loc[gdf[id_column].duplicated(keep=False), id_column].tolist()
        raise ValueError(
            f"ID column {id_column!r} contains duplicate values: {dupes[:10]}"
        )

    target_epsg = f"EPSG:{target_crs}"
    if gdf.crs is None:
        gdf = gdf.set_crs(target_epsg)
    elif gdf.crs.to_epsg() != target_crs:
        gdf = gdf.to_crs(target_epsg)

    return gdf


def get_bounds(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Return the total bounding box of a GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame (any CRS).

    Returns
    -------
    tuple[float, float, float, float]
        ``(minx, miny, maxx, maxy)`` in the CRS of *gdf*.
    """
    return tuple(gdf.total_bounds)  # type: ignore[return-value]
