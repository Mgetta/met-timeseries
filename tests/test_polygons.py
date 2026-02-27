"""
Tests for met_timeseries.polygons — load_polygons dissolve behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping


def _make_geojson(tmp_path: Path, features: list[dict]) -> Path:
    fc = {"type": "FeatureCollection", "features": features}
    path = tmp_path / "catchments.geojson"
    path.write_text(json.dumps(fc))
    return path


def _box_feature(minx, miny, maxx, maxy, **props) -> dict:
    return {
        "type": "Feature",
        "geometry": mapping(box(minx, miny, maxx, maxy)),
        "properties": props,
    }


# ---------------------------------------------------------------------------
# Basic dissolve
# ---------------------------------------------------------------------------

class TestLoadPolygons:
    def test_dissolve_reduces_row_count(self, catchment_geojson: Path) -> None:
        """6 catchments → 3 metzones after dissolve."""
        from met_timeseries.polygons import load_polygons

        gdf = load_polygons(str(catchment_geojson), metzone_column="metzone_id")
        assert len(gdf) == 3

    def test_output_columns(self, catchment_geojson: Path) -> None:
        """Result must contain exactly [metzone_id, geometry]."""
        from met_timeseries.polygons import load_polygons

        gdf = load_polygons(str(catchment_geojson), metzone_column="metzone_id")
        assert set(gdf.columns) == {"metzone_id", "geometry"}

    def test_metzone_ids_preserved(self, catchment_geojson: Path) -> None:
        from met_timeseries.polygons import load_polygons

        gdf = load_polygons(str(catchment_geojson), metzone_column="metzone_id")
        assert set(gdf["metzone_id"]) == {"MZ_A", "MZ_B", "MZ_C"}

    def test_reprojection(self, catchment_geojson: Path) -> None:
        from met_timeseries.polygons import load_polygons

        gdf = load_polygons(
            str(catchment_geojson), metzone_column="metzone_id", target_crs="EPSG:4326"
        )
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_custom_target_crs(self, catchment_geojson: Path) -> None:
        from met_timeseries.polygons import load_polygons

        gdf = load_polygons(
            str(catchment_geojson), metzone_column="metzone_id", target_crs="EPSG:32612"
        )
        assert gdf.crs.to_epsg() == 32612

    def test_dissolve_merges_adjacent_catchments(self, tmp_path: Path) -> None:
        """Two adjacent boxes with the same metzone_id become one polygon."""
        features = [
            _box_feature(0, 0, 1, 1, metzone_id="Z1"),
            _box_feature(1, 0, 2, 1, metzone_id="Z1"),
        ]
        path = _make_geojson(tmp_path, features)
        from met_timeseries.polygons import load_polygons

        gdf = load_polygons(str(path), metzone_column="metzone_id")
        assert len(gdf) == 1
        # Merged area should equal sum of individual areas (approx)
        assert gdf.geometry.iloc[0].area == pytest.approx(2.0, rel=1e-6)

    def test_multiple_metzones_separate(self, tmp_path: Path) -> None:
        features = [
            _box_feature(0, 0, 1, 1, metzone_id="A"),
            _box_feature(2, 0, 3, 1, metzone_id="B"),
        ]
        path = _make_geojson(tmp_path, features)
        from met_timeseries.polygons import load_polygons

        gdf = load_polygons(str(path), metzone_column="metzone_id")
        assert len(gdf) == 2
        assert set(gdf["metzone_id"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# Invalid column
# ---------------------------------------------------------------------------

class TestLoadPolygonsValidation:
    def test_missing_column_raises(self, catchment_geojson: Path) -> None:
        from met_timeseries.polygons import load_polygons

        with pytest.raises(ValueError, match="not found"):
            load_polygons(str(catchment_geojson), metzone_column="nonexistent_col")

    def test_error_message_shows_available_columns(self, catchment_geojson: Path) -> None:
        from met_timeseries.polygons import load_polygons

        with pytest.raises(ValueError, match="metzone_id"):
            load_polygons(str(catchment_geojson), metzone_column="bad_col")


# ---------------------------------------------------------------------------
# Invalid geometry handling
# ---------------------------------------------------------------------------

class TestInvalidGeometry:
    def test_invalid_geometry_fixed(self, tmp_path: Path) -> None:
        """A self-intersecting polygon should be fixed with .buffer(0)."""
        from shapely.geometry import Polygon
        from met_timeseries.polygons import load_polygons

        # Figure-8 / bowtie polygon (self-intersecting → invalid)
        bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        assert not bowtie.is_valid

        features = [
            {
                "type": "Feature",
                "geometry": mapping(bowtie),
                "properties": {"metzone_id": "BT"},
            }
        ]
        path = _make_geojson(tmp_path, features)
        gdf = load_polygons(str(path), metzone_column="metzone_id")
        assert gdf.geometry.is_valid.all()
