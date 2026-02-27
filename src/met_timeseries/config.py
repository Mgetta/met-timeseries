"""
Pipeline configuration dataclass.

Users construct a :class:`PipelineConfig` and pass it to :func:`run_pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the met-timeseries pipeline.

    Parameters
    ----------
    catchment_path:
        Path to a vector file (shapefile, GeoPackage, GeoJSON, Parquet, or any
        format supported by GeoPandas) containing individual model catchments.
        Each feature must have an attribute column that identifies the metzone
        it belongs to (see *metzone_column*).
    metzone_column:
        Name of the column in *catchment_path* that groups catchments into
        metzones.  All catchments sharing the same value in this column are
        dissolved into a single analysis polygon.  Defaults to ``"metzone_id"``.
    output_dir:
        Directory where output Parquet files are written.
    cache_dir:
        Directory used to cache downloaded source files (NLDAS, PRISM, …).
    start_year:
        First calendar year to process (inclusive).
    end_year:
        Last calendar year to process (inclusive).
    variables:
        List of variable names to derive and output.
    target_crs:
        EPSG code or WKT string for the coordinate reference system used
        internally.  Input catchments are re-projected to this CRS after
        dissolving.
    ledger_path:
        Path to the CSV ledger file that tracks which month/step combinations
        have been completed.
    """

    catchment_path: str
    metzone_column: str = "metzone_id"
    output_dir: str = "output"
    cache_dir: str = ".cache"
    start_year: int = 2000
    end_year: int = 2020
    variables: list[str] = field(default_factory=lambda: ["precip", "tmax", "tmin"])
    target_crs: str = "EPSG:4326"
    ledger_path: str = "ledger.csv"

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
