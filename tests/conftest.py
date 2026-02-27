"""Shared pytest fixtures for met-timeseries tests."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import shape


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def sample_geojson_path(tmp_path: Path) -> Path:
    """Write a minimal GeoJSON with 3 polygons and return its path."""
    features = [
        {
            "type": "Feature",
            "properties": {"polygon_id": f"poly_{i}"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-84.0 + i, 35.0],
                    [-83.5 + i, 35.0],
                    [-83.5 + i, 35.5],
                    [-84.0 + i, 35.5],
                    [-84.0 + i, 35.0],
                ]],
            },
        }
        for i in range(3)
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    path = tmp_path / "polygons.geojson"
    path.write_text(json.dumps(geojson))
    return path


@pytest.fixture
def sample_gdf(sample_geojson_path: Path) -> gpd.GeoDataFrame:
    """Return a loaded GeoDataFrame from the sample GeoJSON."""
    from met_timeseries.polygons import load_polygons
    return load_polygons(sample_geojson_path)
