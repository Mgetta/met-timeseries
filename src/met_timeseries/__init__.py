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
from met_timeseries.pipeline import run_pipeline
from met_timeseries.polygons import load_polygons

__all__ = ["PipelineConfig", "run_pipeline", "load_polygons"]
