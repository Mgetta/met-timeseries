import numpy as np
import pytest

from met_timeseries.spatial.weights import _calculate_raster_coverage, compute_weights
from tests.conftest import make_polygon

import sys
import os

# Get the absolute path of the directory this file is in (tests/unit/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths you need
tests_dir = os.path.abspath(os.path.join(current_dir, '..'))
root_dir = os.path.abspath(os.path.join(tests_dir, '..'))

# Inject them into Python's search path (just like Pytest does)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

def _lat_lon_tuples(grid):
    return (
        tuple(float(v) for v in grid.lat.values),
        tuple(float(v) for v in grid.lon.values),
    )


def test_compute_weights_for_offset_inside_polygon(
        grid_20x30,
):

    polygon, _ = make_polygon(
        row_start=4.25,
        row_stop=5.25,
        col_start=4.25,
        col_stop=6.25,
    )
    lats, lons = _lat_lon_tuples(grid_20x30)
    weights = compute_weights(polygon, lats, lons, normalize=False)

    # all grid cells that are partially covered by the polygon should have a weight > 0
    assert np.count_nonzero(weights) == 6
    assert weights.sum() == pytest.approx(2.0)
    assert weights.shape == (grid_20x30.sizes["lat"], grid_20x30.sizes["lon"])

    #The kernel computes area ratios in equal-area projection. 
    # Longitude splits are exact (6933 is linear in longitude), 
    # but latitude splits are not: the projection's latitude scaling varies as cos(lat), 
    # about 0.17% across a 0.1° cell at 45°N, 
    # so a nominal 0.25 latitude fraction is really 0.25 ± ~2×10⁻⁴.
    # Therefore using a higher tolerance for computed area ratios, and using pytest.approx for comparison.


    assert weights[4, 4] == pytest.approx(9/16, abs = 1e-3)
    assert weights[4,5] == pytest.approx(12/16, abs = 1e-3)
    assert weights[4,6] == pytest.approx(3/16, abs = 1e-3)
    assert weights[5,4] == pytest.approx(3/16, abs = 1e-3)
    assert weights[5,5] == pytest.approx(4/16, abs = 1e-3)
    assert weights[5,6] == pytest.approx(1/16, abs = 1e-3)


def test_compute_weights_for_single_offset_inside_polygon(
        grid_20x30,
):

    polygon, _ = make_polygon(
        row_start=4.5,
        row_stop=5.5,
        col_start=4.5,
        col_stop=5.5,
    )
    lats, lons = _lat_lon_tuples(grid_20x30)
    weights = compute_weights(polygon, lats, lons, normalize=False)

    # all grid cells that are partially covered by the polygon should have a weight > 0
    assert np.count_nonzero(weights) == 4
    assert weights.sum() == pytest.approx(1.0)

    #The kernel computes area ratios in equal-area projection. 
    # Longitude splits are exact (6933 is linear in longitude), 
    # but latitude splits are not: the projection's latitude scaling varies as cos(lat), 
    # about 0.17% across a 0.1° cell at 45°N, 
    # so a nominal 0.25 latitude fraction is really 0.25 ± ~2×10⁻⁴.
    # Therefore using a higher tolerance for computed area ratios, and using pytest.approx for comparison.
    assert weights[4, 4] == pytest.approx(1/4, abs = 1e-3)
    assert weights[4,5] == pytest.approx(1/4, abs = 1e-3)
    assert weights[5,4] == pytest.approx(1/4, abs = 1e-3)
    assert weights[5,5] == pytest.approx(1/4, abs = 1e-3)


def test_calculate_raster_coverage_for_inside_polygon(grid_20x30, overlapping_polygon):
    
    polygon, _ = make_polygon(
        row_start=4.5,
        row_stop=5.5,
        col_start=4.5,
        col_stop=5.5,
    )
    lats, lons = _lat_lon_tuples(grid_20x30)

    coverage = _calculate_raster_coverage(polygon, lats, lons)
    assert coverage == pytest.approx(1.0)
  
def test_compute_weights_for_outside_polygon(grid_20x30, outside_polygon):

    polygon, _ = make_polygon(
        row_start=-1,
        row_stop=1,
        col_start=-1,
        col_stop=1,
    )
    lats, lons = _lat_lon_tuples(grid_20x30)

    coverage = _calculate_raster_coverage(polygon, lats, lons)
    assert coverage == pytest.approx(.25,abs = 1e-3)
