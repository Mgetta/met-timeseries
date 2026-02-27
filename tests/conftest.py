"""
Shared pytest fixtures for the met-timeseries test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Vector / polygon fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def catchment_geojson(tmp_path: Path) -> Path:
    """GeoJSON with 6 catchments across 3 metzones (2 catchments per metzone)."""
    features = []
    metzone_ids = ["MZ_A", "MZ_A", "MZ_B", "MZ_B", "MZ_C", "MZ_C"]
    # 6 small adjacent boxes arranged in a 2×3 grid
    coords = [
        (-110.0, 45.0, -109.5, 45.5),
        (-109.5, 45.0, -109.0, 45.5),
        (-110.0, 45.5, -109.5, 46.0),
        (-109.5, 45.5, -109.0, 46.0),
        (-110.0, 46.0, -109.5, 46.5),
        (-109.5, 46.0, -109.0, 46.5),
    ]
    for mz, (minx, miny, maxx, maxy) in zip(metzone_ids, coords):
        geom = box(minx, miny, maxx, maxy)
        features.append(
            {
                "type": "Feature",
                "geometry": geom.__geo_interface__,
                "properties": {"metzone_id": mz, "catchment_name": f"catch_{mz}"},
            }
        )
    fc = {"type": "FeatureCollection", "features": features}
    path = tmp_path / "catchments.geojson"
    path.write_text(json.dumps(fc))
    return path


@pytest.fixture()
def dissolved_polygons(catchment_geojson: Path) -> gpd.GeoDataFrame:
    """Pre-dissolved GeoDataFrame with 3 metzone polygons."""
    from met_timeseries.polygons import load_polygons

    return load_polygons(str(catchment_geojson), metzone_column="metzone_id")


# ---------------------------------------------------------------------------
# Ledger fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ledger_path(tmp_path: Path) -> Path:
    """Path to a fresh (non-existent) ledger CSV."""
    return tmp_path / "ledger.csv"


# ---------------------------------------------------------------------------
# Raster / Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_dataset() -> xr.Dataset:
    """Tiny 4×4 grid xarray Dataset with 'APCP' and 'TMP' variables."""
    lats = np.linspace(45.1, 45.4, 4)
    lons = np.linspace(-109.9, -109.6, 4)
    data = np.ones((4, 4))
    return xr.Dataset(
        {
            "APCP": xr.DataArray(data * 5.0, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}),
            "TMP": xr.DataArray(data * 280.0, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}),
        }
    )


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_config(catchment_geojson: Path, tmp_path: Path):
    """Minimal PipelineConfig pointing at tmp_path directories."""
    from met_timeseries.config import PipelineConfig

    return PipelineConfig(
        catchment_path=str(catchment_geojson),
        metzone_column="metzone_id",
        output_dir=str(tmp_path / "output"),
        cache_dir=str(tmp_path / "cache"),
        start_year=2000,
        end_year=2000,
        ledger_path=str(tmp_path / "ledger.csv"),
    )
