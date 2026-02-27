"""Tests for polygon loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from met_timeseries.polygons import load_polygons, get_bounds


def test_load_polygons_basic(sample_geojson_path: Path):
    gdf = load_polygons(sample_geojson_path)
    assert len(gdf) == 3
    assert "polygon_id" in gdf.columns
    assert gdf.crs.to_epsg() == 4326


def test_load_polygons_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_polygons("/nonexistent/path.geojson")


def test_load_polygons_missing_id_column(sample_geojson_path: Path):
    with pytest.raises(ValueError, match="ID column"):
        load_polygons(sample_geojson_path, id_column="nonexistent_col")


def test_load_polygons_duplicate_ids(tmp_path: Path):
    features = [
        {
            "type": "Feature",
            "properties": {"polygon_id": "dup"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-84, 35], [-83, 35], [-83, 36], [-84, 36], [-84, 35]]],
            },
        },
        {
            "type": "Feature",
            "properties": {"polygon_id": "dup"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-82, 35], [-81, 35], [-81, 36], [-82, 36], [-82, 35]]],
            },
        },
    ]
    path = tmp_path / "dup.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    with pytest.raises(ValueError, match="duplicate"):
        load_polygons(path)


def test_get_bounds(sample_gdf):
    bounds = get_bounds(sample_gdf)
    assert len(bounds) == 4
    minx, miny, maxx, maxy = bounds
    assert minx < maxx
    assert miny < maxy
