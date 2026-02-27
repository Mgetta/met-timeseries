"""Tests for polygon aggregation."""

from __future__ import annotations

import numpy as np
import pytest


def test_aggregate_import():
    """Aggregation module should import without error."""
    from met_timeseries.aggregation import aggregate_to_polygons
    assert callable(aggregate_to_polygons)


def test_aggregate_requires_exactextract(monkeypatch, sample_gdf):
    """Should raise ImportError if exactextract is not installed."""
    import sys
    import importlib

    from met_timeseries import aggregation

    # Temporarily make exact_extract unavailable
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def mock_import(name, *args, **kwargs):
        if name == "exactextract":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ImportError, match="exactextract"):
        from met_timeseries.aggregation import aggregate_to_polygons
        import xarray as xr
        raster = xr.DataArray(
            np.ones((5, 5)),
            coords={"lat": np.linspace(34, 36, 5), "lon": np.linspace(-85, -83, 5)},
            dims=["lat", "lon"],
        )
        aggregate_to_polygons(raster, sample_gdf)
