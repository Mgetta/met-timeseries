from met_timeseries.geometry import BoundingBox, CACHE_BOUNDS, bounds_to_polygon, clip_dataset
import numpy as np
import pytest

def test_bounds_to_polygon():
    polygon = bounds_to_polygon(CACHE_BOUNDS)
    assert polygon.bounds == (CACHE_BOUNDS.west, CACHE_BOUNDS.south, CACHE_BOUNDS.east, CACHE_BOUNDS.north)


def _coord_present(values, expected):
    return bool(np.any(np.isclose(values, expected)))


@pytest.mark.parametrize("grid_fixture", ["grid_20x30", "descending_grid_20x30"])
def test_clip_dataset_includes_cells_whose_edges_overlap_bounds(request, grid_fixture):
    grid = request.getfixturevalue(grid_fixture)
    sorted_lats = np.sort(grid.lat.values)
    lons = grid.lon.values
    dlat = sorted_lats[1] - sorted_lats[0]
    dlon = lons[1] - lons[0]

    bounding_box = BoundingBox(
        west=float(lons[5] + dlon * 0.25),
        south=float(sorted_lats[5] + dlat * 0.25),
        east=float(lons[10] - dlon * 0.25),
        north=float(sorted_lats[10] - dlat * 0.25),
    )

    clipped = clip_dataset(grid, bounding_box)
    assert _coord_present(clipped.lon.values, lons[5])
    assert _coord_present(clipped.lon.values, lons[10])
    assert _coord_present(clipped.lat.values, sorted_lats[5])
    assert _coord_present(clipped.lat.values, sorted_lats[10])
    assert not _coord_present(clipped.lon.values, lons[4])
    assert not _coord_present(clipped.lon.values, lons[11])

def test_BoundingBox_contains():
    bounding_box = BoundingBox(
        west=-95,
        south=43,
        east=-91,
        north=50
    )
    assert CACHE_BOUNDS.contains(bounding_box)