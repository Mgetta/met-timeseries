"""Variable derivation functions for met-timeseries.

Each module exposes a single public function that computes a derived
meteorological variable from raw NLDAS-2 (or PRISM) data arrays.
"""

from met_timeseries.derivations.cloud_cover import compute_cloud_cover
from met_timeseries.derivations.dewpoint import compute_dewpoint
from met_timeseries.derivations.disaggregation import disaggregate_precip
from met_timeseries.derivations.pet import compute_pet_fao56
from met_timeseries.derivations.wind import compute_wind_speed

__all__ = [
    "compute_wind_speed",
    "compute_dewpoint",
    "compute_cloud_cover",
    "compute_pet_fao56",
    "disaggregate_precip",
]
