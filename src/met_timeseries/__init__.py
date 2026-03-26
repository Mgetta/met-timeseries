"""
met-timeseries: meteorological timeseries downloading and processing for hydrologic models.

Procedural API:
    from met_timeseries.config import PipelineConfig
    from met_timeseries.pipeline import run_pipeline

    config = PipelineConfig(
        catchment_path="catchments.gpkg",
        metzone_column="metzone_id",
        output_dir="output",
        start_year=2000,
        end_year=2005,
    )
    run_pipeline(config)
"""

from met_timeseries.config import PipelineConfig
from met_timeseries.polygons import load_polygons

# pipeline.py still references the old derive_variables API; guard the import
# so that the rest of the package remains usable while the pipeline wiring is
# updated separately.
try:
    from met_timeseries.pipeline import run_pipeline
except ImportError:
    run_pipeline = None  # type: ignore[assignment,misc]

__all__ = ["PipelineConfig", "run_pipeline", "load_polygons"]
