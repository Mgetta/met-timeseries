"""met-timeseries: Hourly meteorological timeseries extraction and aggregation.

This package provides a pipeline for extracting hourly meteorological timeseries
from gridded data products (NLDAS-2, PRISM, etc.) and aggregating them to polygon
areas of interest using area-weighted zonal statistics.
"""

from met_timeseries.config import PipelineConfig
from met_timeseries.pipeline import Pipeline

__version__ = "0.1.0"
__all__ = ["Pipeline", "PipelineConfig", "__version__"]
