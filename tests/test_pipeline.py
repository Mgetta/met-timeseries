"""Tests for the Pipeline class."""

from __future__ import annotations

from pathlib import Path

import pytest

from met_timeseries.config import PipelineConfig
from met_timeseries.ledger import Ledger
from met_timeseries.pipeline import Pipeline


def test_pipeline_skips_complete_steps(tmp_output_dir: Path, sample_geojson_path: Path):
    """Pipeline should skip months that are already in the ledger."""
    config = PipelineConfig(
        polygon_path=sample_geojson_path,
        output_dir=tmp_output_dir,
        start_date="2020-01-01",
        end_date="2020-01-31",
    )
    ledger = Ledger(tmp_output_dir)
    # Pre-mark all steps complete for every month in the year
    for step in ["nldas", "prism", "derive", "aggregate"]:
        for month in range(1, 13):
            ledger.mark_complete(step, 2020, month)

    pipeline = Pipeline(config)

    # Override _run_step to detect if it's called
    called_steps = []

    def mock_run_step(step, year, month, polygons):
        called_steps.append((step, year, month))

    pipeline._run_step = mock_run_step

    pipeline.run()
    assert called_steps == [], "No steps should run if all are complete"


def test_pipeline_initialises(tmp_output_dir: Path, sample_geojson_path: Path):
    config = PipelineConfig(
        polygon_path=sample_geojson_path,
        output_dir=tmp_output_dir,
        start_date="2020-01-01",
        end_date="2020-01-31",
    )
    pipeline = Pipeline(config)
    assert isinstance(pipeline.ledger, Ledger)
    assert pipeline.config is config
