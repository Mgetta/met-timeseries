from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry


TEST_GRID_N_LAT = 20
TEST_GRID_N_LON = 30
TEST_GRID_WEST = -96.0
TEST_GRID_SOUTH = 44.0
TEST_GRID_DLAT = 0.1
TEST_GRID_DLON = 0.1

def make_grid(
    *,
    n_lat: int = TEST_GRID_N_LAT,
    n_lon: int = TEST_GRID_N_LON,
    west: float = TEST_GRID_WEST,
    south: float = TEST_GRID_SOUTH,
    dlat: float = TEST_GRID_DLAT,
    dlon: float = TEST_GRID_DLON,
    descending_lat: bool = False,
    with_time: bool = False,
    periods: int = 3,
    freq: str = "1h",
    start: str = "2000-01-01",
    var_name: str = "value",
) -> xr.Dataset:
    """Build a small regular lat/lon grid with predictable values."""
    lat = south + dlat * (np.arange(n_lat, dtype=float) + 0.5)
    lon = west + dlon * (np.arange(n_lon, dtype=float) + 0.5)
    if descending_lat:
        lat = lat[::-1]

    row_values = np.arange(n_lat, dtype=float)[:, np.newaxis]
    col_values = np.arange(n_lon, dtype=float)[np.newaxis, :]
    data = row_values * 100.0 + col_values

    coords: dict[str, object] = {"lat": lat, "lon": lon}
    dims: tuple[str, ...] = ("lat", "lon")
    if with_time:
        time = pd.date_range(start=start, periods=periods, freq=freq)
        coords["time"] = time
        dims = ("time", "lat", "lon")
        data = (
            data[np.newaxis, :, :]
            + (
                np.arange(periods, dtype=float)[:, np.newaxis, np.newaxis]
                * 10_000.0
            )
        )

    return xr.Dataset({var_name: (dims, data)}, coords=coords)



def make_polygon(
    *,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    west: float = TEST_GRID_WEST,
    south: float = TEST_GRID_SOUTH,
    dlat: float = TEST_GRID_DLAT,
    dlon: float = TEST_GRID_DLON,
    n_lat: int = TEST_GRID_N_LAT,
    n_lon: int = TEST_GRID_N_LON,
    shape: str = "box",
) -> tuple[BaseGeometry, int]:
    """Build a polygon and count default-grid cells with nonzero area overlap."""
    if row_stop <= row_start:
        raise ValueError("row_stop must be greater than row_start.")
    if col_stop <= col_start:
        raise ValueError("col_stop must be greater than col_start.")

    bounds = (
        west + col_start * dlon,
        south + row_start * dlat,
        west + col_stop * dlon,
        south + row_stop * dlat,
    )

    polygon = shapely_box(*bounds)
    if shape == "blob":
        polygon = polygon.buffer(min(dlon, dlat) * 0.65, resolution=8)
    elif shape != "box":
        raise ValueError("shape must be one of 'box' or 'blob'.")

    grid_count = _count_overlapping_grid_cells(
        polygon,
        n_lat=n_lat,
        n_lon=n_lon,
        west=west,
        south=south,
        dlat=dlat,
        dlon=dlon,
    )
    return polygon, grid_count


@pytest.fixture()
def grid_20x30() -> xr.Dataset:
    """Regular 20-by-30 ascending-latitude grid."""
    return make_grid()


@pytest.fixture()
def descending_grid_20x30() -> xr.Dataset:
    """Regular 20-by-30 grid with descending latitude coordinates."""
    return make_grid(descending_lat=True)


@pytest.fixture()
def overlapping_grid_10x15() -> xr.Dataset:
    """A 10-by-15 grid with 80 percent of its width inside the default grid."""
    east = TEST_GRID_WEST + TEST_GRID_N_LON * TEST_GRID_DLON
    width = 15 * TEST_GRID_DLON
    return make_grid(
        n_lat=10,
        n_lon=15,
        west=east - width * 0.8,
        south=TEST_GRID_SOUTH + 2 * TEST_GRID_DLAT,
        dlat=TEST_GRID_DLAT,
        dlon=TEST_GRID_DLON,
    )


@pytest.fixture()
def inside_polygon -> tuple[BaseGeometry, int]:
    """Cell-edge-aligned interior polygon and its covered grid-cell count."""
    row_start = 4
    row_stop = 12
    col_start = 5
    col_stop = 15
    return make_polygon(
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        shape="box",
    )


@pytest.fixture()
def inside_blob_polygon() -> tuple[BaseGeometry, int]:
    """Irregular interior polygon and its covered grid-cell count."""
    row_start = 4
    row_stop = 12
    col_start = 5
    col_stop = 15
    return make_polygon(
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        shape="blob",
    )


@pytest.fixture()
def overlapping_polygon() -> tuple[BaseGeometry, int]:
    """Partially overlapping polygon and its covered grid-cell count."""
    row_start = 5
    row_stop = 13
    col_start = TEST_GRID_N_LON - 6
    col_stop = TEST_GRID_N_LON + 4
    return make_polygon(
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        shape="box",
    )


@pytest.fixture()
def outside_polygon() -> tuple[BaseGeometry, int]:
    """Outside polygon and its covered grid-cell count."""
    row_start = 2
    row_stop = 8
    col_start = TEST_GRID_N_LON + 2
    col_stop = TEST_GRID_N_LON + 8
    return make_polygon(
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        shape="box",
    )


@pytest.fixture()
def catchment_gdf() -> gpd.GeoDataFrame:
    """Simple catchments with two geometries in metzone A and one in B."""
    zone_a_row_start = 4
    zone_a_row_stop = 12
    zone_a_col_start = 5
    zone_a_col_stop = 15
    zone_a_mid_col = (zone_a_col_start + zone_a_col_stop) // 2

    zone_b_row_start = 5
    zone_b_row_stop = 13
    zone_b_col_start = TEST_GRID_N_LON - 6
    zone_b_col_stop = TEST_GRID_N_LON + 4

    zone_a_west, _zone_a_west_count = make_polygon(
        row_start=zone_a_row_start,
        row_stop=zone_a_row_stop,
        col_start=zone_a_col_start,
        col_stop=zone_a_mid_col,
    )
    zone_a_east, _zone_a_east_count = make_polygon(
        row_start=zone_a_row_start,
        row_stop=zone_a_row_stop,
        col_start=zone_a_mid_col,
        col_stop=zone_a_col_stop,
    )
    zone_b, _zone_b_count = make_polygon(
        row_start=zone_b_row_start,
        row_stop=zone_b_row_stop,
        col_start=zone_b_col_start,
        col_stop=zone_b_col_stop,
    )

    return gpd.GeoDataFrame(
        {
            "metzone": ["A", "A", "B"],
            "geometry": [
                zone_a_west,
                zone_a_east,
                zone_b,
            ],
        },
        crs="EPSG:4326",
    )