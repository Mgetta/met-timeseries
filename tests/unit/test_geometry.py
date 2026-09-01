import pytest
from pathlib import Path
from met_timeseries.geometry import BoundingBox, CACHE_BOUNDS, bounds_to_polygon, clip_dataset
import xarray as xr

def test_bounds_to_polygon():
    polygon = bounds_to_polygon(CACHE_BOUNDS)
    assert polygon.bounds == (CACHE_BOUNDS.west, CACHE_BOUNDS.south, CACHE_BOUNDS.east, CACHE_BOUNDS.north)

def generate_xarray_grid():
    lats = list(range(int(CACHE_BOUNDS.south), int(CACHE_BOUNDS.north) + 1),)
    lons = list(range(int(CACHE_BOUNDS.west), int(CACHE_BOUNDS.east) + 1))
    da = xr.DataArray(
        1,
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"]
    )
    return da


def test_clip_dataset():
    # Create a simple test dataset
    da = generate_xarray_grid()

    bounding_box = BoundingBox(
        west=da.lon.min().item() + 2,
        south=da.lat.min().item() + 2,
        east=da.lon.max().item() - 2,
        north=da.lat.max().item() - 2
    )

    clipped_da = clip_dataset(da, bounding_box)
    assert clipped_da.lat.min() >= bounding_box.south
    assert clipped_da.lat.max() <= bounding_box.north
    assert clipped_da.lon.min() >= bounding_box.west
    assert clipped_da.lon.max() <= bounding_box.east

def test_BoundingBox_contains():
    bounding_box = BoundingBox(
        west=-95,
        south=43,
        east=-91,
        north=50
    )
    assert CACHE_BOUNDS.contains(bounding_box)