"""
Tests for the procedural pipeline API.

These tests mock out the data-fetching layer so they run without network access
or valid NASA/PRISM credentials.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd


def _make_mock_dataset(lats, lons, n_times: int = 3) -> xr.Dataset:
    """Return a tiny Dataset with APCP and TMP for mocking fetch_nldas.

    The dataset has a ``time`` dimension to simulate hourly output.
    """
    times = pd.date_range("2000-01-01", periods=n_times, freq="h")
    data = np.ones((n_times, len(lats), len(lons)))
    return xr.Dataset(
        {
            "APCP": xr.DataArray(
                data * 10.0,
                dims=["time", "lat", "lon"],
                coords={"time": times, "lat": lats, "lon": lons},
            ),
            "TMP": xr.DataArray(
                data * 285.0,
                dims=["time", "lat", "lon"],
                coords={"time": times, "lat": lats, "lon": lons},
            ),
        }
    )


class TestAssembleFinalOutput:
    def test_merges_on_metzone_id(self) -> None:
        from met_timeseries.pipeline import assemble_final_output

        nldas_df = pd.DataFrame({"metzone_id": ["A", "B"], "precip_mm": [100.0, 200.0]})
        prism_df = pd.DataFrame({"metzone_id": ["A", "B"], "ppt": [95.0, 195.0]})

        result = assemble_final_output(nldas_df, prism_df, 2000, 1)
        assert "metzone_id" in result.columns
        assert "precip_mm" in result.columns
        assert "ppt" in result.columns
        assert result["year"].iloc[0] == 2000
        assert result["month"].iloc[0] == 1

    def test_outer_join_fills_missing(self) -> None:
        from met_timeseries.pipeline import assemble_final_output

        nldas_df = pd.DataFrame({"metzone_id": ["A"], "precip_mm": [100.0]})
        prism_df = pd.DataFrame({"metzone_id": ["B"], "ppt": [50.0]})

        result = assemble_final_output(nldas_df, prism_df, 2001, 6)
        assert len(result) == 2  # outer join gives 2 rows


class TestProcessNldasMonth:
    def test_marks_ledger_complete(self, dissolved_polygons: gpd.GeoDataFrame, tmp_path: Path) -> None:
        """After a successful run the ledger entry should be present."""
        from met_timeseries.pipeline import process_nldas_month
        from met_timeseries.ledger import is_complete

        lats = np.linspace(45.0, 45.5, 4)
        lons = np.linspace(-110.0, -109.0, 4)
        mock_ds = _make_mock_dataset(lats, lons)

        ledger = str(tmp_path / "ledger.csv")

        with patch("met_timeseries.pipeline.fetch_nldas", return_value=mock_ds):
            process_nldas_month(
                polygons=dissolved_polygons,
                metzone_column="metzone_id",
                year=2000,
                month=1,
                output_dir=str(tmp_path / "output"),
                cache_dir=str(tmp_path / "cache"),
                variables=["APCP", "TMP"],
                ledger_path=ledger,
            )

        assert is_complete(ledger, "nldas", 2000, 1)

    def test_skips_already_complete(self, dissolved_polygons: gpd.GeoDataFrame, tmp_path: Path) -> None:
        """When the ledger shows completion, fetch_nldas should not be called."""
        from met_timeseries.pipeline import process_nldas_month
        from met_timeseries.ledger import mark_complete

        ledger = str(tmp_path / "ledger.csv")
        mark_complete(ledger, "nldas", 2000, 1)

        with patch("met_timeseries.pipeline.fetch_nldas") as mock_fetch:
            process_nldas_month(
                polygons=dissolved_polygons,
                metzone_column="metzone_id",
                year=2000,
                month=1,
                output_dir=str(tmp_path / "output"),
                cache_dir=str(tmp_path / "cache"),
                variables=["APCP"],
                ledger_path=ledger,
            )
        mock_fetch.assert_not_called()

    def test_saves_parquet_per_metzone(self, dissolved_polygons: gpd.GeoDataFrame, tmp_path: Path) -> None:
        """One Parquet file should be written per (metzone, variable)."""
        from met_timeseries.pipeline import process_nldas_month

        lats = np.linspace(45.0, 46.5, 8)
        lons = np.linspace(-110.0, -109.0, 8)
        mock_ds = _make_mock_dataset(lats, lons)

        output_dir = tmp_path / "output"
        ledger = str(tmp_path / "ledger.csv")

        with patch("met_timeseries.pipeline.fetch_nldas", return_value=mock_ds):
            process_nldas_month(
                polygons=dissolved_polygons,
                metzone_column="metzone_id",
                year=2000,
                month=3,
                output_dir=str(output_dir),
                cache_dir=str(tmp_path / "cache"),
                variables=["APCP", "TMP"],
                ledger_path=ledger,
            )

        parquet_files = list(output_dir.rglob("*.parquet"))
        # 3 metzones × 2 derived variables = 6 files
        assert len(parquet_files) == 6

    def test_parquet_contains_metzone_id(self, dissolved_polygons: gpd.GeoDataFrame, tmp_path: Path) -> None:
        from met_timeseries.pipeline import process_nldas_month

        lats = np.linspace(45.0, 46.5, 8)
        lons = np.linspace(-110.0, -109.0, 8)
        mock_ds = _make_mock_dataset(lats, lons)

        output_dir = tmp_path / "output"
        ledger = str(tmp_path / "ledger.csv")

        with patch("met_timeseries.pipeline.fetch_nldas", return_value=mock_ds):
            process_nldas_month(
                polygons=dissolved_polygons,
                metzone_column="metzone_id",
                year=2000,
                month=4,
                output_dir=str(output_dir),
                cache_dir=str(tmp_path / "cache"),
                variables=["APCP"],
                ledger_path=ledger,
            )

        for pq in output_dir.rglob("*.parquet"):
            df = pd.read_parquet(pq)
            assert "metzone_id" in df.columns

    def test_parquet_has_hourly_rows(self, dissolved_polygons: gpd.GeoDataFrame, tmp_path: Path) -> None:
        """Each Parquet file should have one row per timestep with a datetime column."""
        from met_timeseries.pipeline import process_nldas_month

        n_times = 5
        lats = np.linspace(45.0, 46.5, 8)
        lons = np.linspace(-110.0, -109.0, 8)
        mock_ds = _make_mock_dataset(lats, lons, n_times=n_times)

        output_dir = tmp_path / "output"
        ledger = str(tmp_path / "ledger.csv")

        with patch("met_timeseries.pipeline.fetch_nldas", return_value=mock_ds):
            process_nldas_month(
                polygons=dissolved_polygons,
                metzone_column="metzone_id",
                year=2000,
                month=5,
                output_dir=str(output_dir),
                cache_dir=str(tmp_path / "cache"),
                variables=["APCP"],
                ledger_path=ledger,
            )

        for pq in output_dir.rglob("*.parquet"):
            df = pd.read_parquet(pq)
            assert len(df) == n_times, f"Expected {n_times} rows, got {len(df)}"
            assert "datetime" in df.columns
            assert df["datetime"].is_monotonic_increasing


class TestRunPipeline:
    def test_run_pipeline_calls_process_nldas(self, pipeline_config, tmp_path: Path) -> None:
        """run_pipeline should attempt to process each incomplete month."""
        from met_timeseries.pipeline import run_pipeline

        lats = np.linspace(45.0, 46.5, 8)
        lons = np.linspace(-110.0, -109.0, 8)
        mock_ds = _make_mock_dataset(lats, lons)

        with patch("met_timeseries.pipeline.fetch_nldas", return_value=mock_ds):
            run_pipeline(pipeline_config)

        # All 12 months of 2000 should now be complete
        from met_timeseries.ledger import is_complete

        for month in range(1, 13):
            assert is_complete(pipeline_config.ledger_path, "nldas", 2000, month)
