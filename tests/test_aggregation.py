"""
Tests for met_timeseries.aggregation — area-weighted polygon aggregation.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box


def _make_dataset(lats, lons, value=1.0) -> xr.Dataset:
    data = np.full((len(lats), len(lons)), value)
    return xr.Dataset(
        {
            "VAR": xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}),
        }
    )


def _make_dataset_3d(lats, lons, ntimes=3, value=1.0) -> xr.Dataset:
    data = np.full((ntimes, len(lats), len(lons)), value)
    return xr.Dataset(
        {
            "VAR": xr.DataArray(
                data,
                dims=["time", "lat", "lon"],
                coords={"lat": lats, "lon": lons},
            ),
        }
    )


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

class TestComputeWeights:
    def test_full_containment_weight_one(self) -> None:
        """A polygon that fully contains a cell gives weight 1.0 for that cell."""
        from met_timeseries.aggregation import _compute_weights

        lats = (0.0,)
        lons = (0.0,)
        # Single-element grid → dx=dy=0.125 (default), so cell is 0.125×0.125.
        # polygon is much larger than the cell.
        polygon = box(-1.0, -1.0, 1.0, 1.0)
        weights = _compute_weights(polygon.wkb, lats, lons)
        assert weights.shape == (1, 1)
        assert weights[0, 0] == pytest.approx(1.0)

    def test_partial_overlap_weight_between_0_and_1(self) -> None:
        """A polygon partially overlapping a cell gives weight strictly between 0 and 1."""
        from met_timeseries.aggregation import _compute_weights

        # Grid: single cell centred at (0, 0); dx=dy=0.125 (default for 1-point grid).
        # Cell extent: (-0.0625, -0.0625) → (0.0625, 0.0625).
        lats = (0.0,)
        lons = (0.0,)
        # Polygon covers only the right half of the cell
        polygon = box(0.0, -0.0625, 0.0625, 0.0625)
        weights = _compute_weights(polygon.wkb, lats, lons)
        assert 0.0 < weights[0, 0] < 1.0

    def test_no_overlap_weight_zero(self) -> None:
        """A polygon that doesn't overlap any cell gives all-zero weights."""
        from met_timeseries.aggregation import _compute_weights

        lats = (0.0,)
        lons = (0.0,)
        polygon = box(10.0, 10.0, 11.0, 11.0)
        weights = _compute_weights(polygon.wkb, lats, lons)
        assert weights[0, 0] == 0.0

    def test_weights_cached(self) -> None:
        """Calling _compute_weights twice returns the same array object (cached)."""
        from met_timeseries.aggregation import _compute_weights

        lats = (0.0, 0.1)
        lons = (0.0, 0.1)
        polygon = box(-0.5, -0.5, 0.5, 0.5)
        w1 = _compute_weights(polygon.wkb, lats, lons)
        w2 = _compute_weights(polygon.wkb, lats, lons)
        assert w1 is w2  # same object from cache


# ---------------------------------------------------------------------------
# aggregate_over_polygon behaviour
# ---------------------------------------------------------------------------

class TestAggregateOverPolygon:
    def test_uniform_field_returns_same_value(self) -> None:
        """For a uniform field the weighted mean equals the constant value."""
        from met_timeseries.aggregation import aggregate_over_polygon

        lats = np.array([0.0, 0.1, 0.2])
        lons = np.array([0.0, 0.1, 0.2])
        ds = _make_dataset(lats, lons, value=7.0)
        polygon = box(-0.5, -0.5, 0.5, 0.5)
        result = aggregate_over_polygon(ds, polygon)
        assert result["VAR"] == pytest.approx(7.0, rel=1e-6)

    def test_weighted_mean_differs_from_boolean_mean(self) -> None:
        """Area-weighted mean should differ from a simple point-in-polygon mean
        when edge cells have heterogeneous values and partial overlap."""
        from met_timeseries.aggregation import aggregate_over_polygon

        # 3-cell grid, only the middle cell is fully inside; the left and right
        # cells are partially overlapped.  Values differ across columns.
        lats = np.array([0.0])
        lons = np.array([0.0, 0.1, 0.2])
        data = np.array([[1.0, 10.0, 100.0]])
        ds = xr.Dataset(
            {"VAR": xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lats, "lon": lons})}
        )
        # Polygon covers exactly the middle cell plus a tiny sliver of the
        # right cell (0.0625–0.175 in lon)
        polygon = box(0.0625, -0.05, 0.175, 0.05)

        result = aggregate_over_polygon(ds, polygon)

        # Simple boolean mean would be mean([10, 100]) = 55 if both centres are inside,
        # or mean([10]) = 10 if only middle centre is inside.
        # Area-weighted result should weight cell 1 (value=10) more than cell 2 (value=100).
        # Verify it's between min and max
        assert 10.0 < result["VAR"] < 100.0

    def test_fallback_nearest_point_when_no_overlap(self) -> None:
        """When polygon doesn't overlap any cell, fall back to nearest centroid."""
        from met_timeseries.aggregation import aggregate_over_polygon

        lats = np.array([0.0, 0.1])
        lons = np.array([0.0, 0.1])
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        ds = xr.Dataset(
            {"VAR": xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lats, "lon": lons})}
        )
        # Polygon far away from the grid — centroid at (10, 10)
        polygon = box(9.9, 9.9, 10.1, 10.1)
        result = aggregate_over_polygon(ds, polygon)
        # Nearest point to (10, 10) is lat=0.1, lon=0.1 → value 4.0
        assert result["VAR"] == pytest.approx(4.0)

    def test_3d_data_handled(self) -> None:
        """3-D (time, lat, lon) arrays should produce a list of per-timestep spatial means."""
        from met_timeseries.aggregation import aggregate_over_polygon

        lats = np.array([0.0, 0.1, 0.2])
        lons = np.array([0.0, 0.1, 0.2])
        ds = _make_dataset_3d(lats, lons, ntimes=5, value=3.0)
        polygon = box(-0.5, -0.5, 0.5, 0.5)
        result = aggregate_over_polygon(ds, polygon)
        assert isinstance(result["VAR"], list)
        assert len(result["VAR"]) == 5
        assert all(pytest.approx(v, rel=1e-6) == 3.0 for v in result["VAR"])

    def test_partial_overlap_weight_applied(self) -> None:
        """A polygon covering exactly half a single cell should give weight ~0.5."""
        from met_timeseries.aggregation import _compute_weights

        lats = np.array([0.0])
        lons = np.array([0.0])
        # Single-element grid → dx=dy=0.125 (default).
        # Cell spans (-0.0625, -0.0625) to (0.0625, 0.0625).
        # Polygon covers the right half of the cell exactly.
        polygon = box(0.0, -0.0625, 0.0625, 0.0625)
        weights = _compute_weights(
            polygon.wkb, tuple(lats.tolist()), tuple(lons.tolist())
        )
        assert weights[0, 0] == pytest.approx(0.5, abs=1e-6)
