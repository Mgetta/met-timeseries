import numpy as np
import pytest

from met_timeseries.spatial.weights import _calculate_raster_coverage, compute_weights


def _lat_lon_tuples(grid):
    return (
        tuple(float(v) for v in grid.lat.values),
        tuple(float(v) for v in grid.lon.values),
    )


def test_compute_weights_for_cell_aligned_inside_polygon(
    grid_20x30,
    inside_polygon
):
    lats, lons = _lat_lon_tuples(grid_20x30)

    weights = compute_weights(inside_polygon[0], lats, lons, normalize=False)

    assert weights.shape == (grid_20x30.sizes["lat"], grid_20x30.sizes["lon"])
    assert np.count_nonzero(weights) == inside_polygon[1]
    assert weights.sum() == pytest.approx(float(inside_polygon[1]))
    assert np.allclose(weights[weights > 0], 1.0)


def test_compute_weights_can_normalize_inside_polygon(
    grid_20x30,
    inside_polygon
):
    lats, lons = _lat_lon_tuples(grid_20x30)

    weights = compute_weights(inside_polygon[0], lats, lons, normalize=True)

    assert weights.sum() == pytest.approx(1.0)
    assert np.count_nonzero(weights) == inside_polygon[1]


def test_compute_weights_for_irregular_inside_blob(
    grid_20x30,
    inside_blob_polygon
):
    lats, lons = _lat_lon_tuples(grid_20x30)

    weights = compute_weights(inside_blob_polygon[0], lats, lons, normalize=False)

    assert weights.shape == (grid_20x30.sizes["lat"], grid_20x30.sizes["lon"])
    assert np.count_nonzero(weights) == inside_blob_polygon[1]
    assert weights.sum() > 0.0
    assert weights.max() <= 1.0 + 1e-12
    assert weights.min() >= -1e-12


def test_calculate_raster_coverage_for_overlapping_polygon(grid_20x30, overlapping_polygon):
    lats, lons = _lat_lon_tuples(grid_20x30)

    coverage = _calculate_raster_coverage(overlapping_polygon[0], lats, lons)
    weights = compute_weights(overlapping_polygon[0], lats, lons, normalize=False)

    assert 0.0 < coverage < 1.0
    assert weights.sum() > 0.0


def test_compute_weights_for_outside_polygon(grid_20x30, outside_polygon):
    lats, lons = _lat_lon_tuples(grid_20x30)

    coverage = _calculate_raster_coverage(outside_polygon[0], lats, lons)
    weights = compute_weights(outside_polygon[0], lats, lons, normalize=False)

    assert coverage == pytest.approx(0.0)
    assert np.count_nonzero(weights) == 0
    assert weights.sum() == pytest.approx(0.0)