"""Tests for PipelineConfig."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from met_timeseries.config import PipelineConfig


def test_defaults():
    config = PipelineConfig(polygon_path="polygons.geojson")
    assert config.output_dir == Path("output")
    assert config.start_date == "1996-01-01"
    assert config.sources == ["nldas", "prism"]
    assert config.polygon_id_column == "polygon_id"
    assert config.crs == 4326


def test_end_date_present_resolves_to_today():
    config = PipelineConfig(polygon_path="p.geojson", end_date="present")
    assert config.end_date == datetime.date.today().isoformat()


def test_invalid_start_date():
    with pytest.raises(ValueError, match="start_date"):
        PipelineConfig(polygon_path="p.geojson", start_date="not-a-date")


def test_start_after_end_raises():
    with pytest.raises(ValueError, match="start_date"):
        PipelineConfig(polygon_path="p.geojson", start_date="2021-01-01", end_date="2020-01-01")


def test_invalid_source():
    with pytest.raises(ValueError, match="Unknown source"):
        PipelineConfig(polygon_path="p.geojson", sources=["invalid"])


def test_year_properties():
    config = PipelineConfig(
        polygon_path="p.geojson", start_date="2020-03-01", end_date="2022-07-15"
    )
    assert config.start_year == 2020
    assert config.end_year == 2022
