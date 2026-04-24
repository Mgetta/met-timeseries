"""
Source-agnostic derived variable calculations.
"""

from met_timeseries.derivations.wind import wind_speed, adjust_wind_height
from met_timeseries.derivations.humidity import (
    dewpoint_from_specific_humidity,
    dewpoint_from_specific_humidity_cc,
    dewpoint_august_roche_magnus,
)
from met_timeseries.derivations.temperature import (
    kelvin_to_fahrenheit,
    kelvin_to_celsius,
)
from met_timeseries.derivations.radiation import (
    cloud_cover_davis,
    thompson_sky_cover_radiation,
    cloud_cover_thompson,
    cloud_cover_linear,
    hamon_weiss_wilson_clearsky,
    hamon_weiss_wilson_actual_radiation,
)
from met_timeseries.derivations.pet import (
    pet_hargreaves,
    pevt_penman_pyet_daily,
    pevt_penman_pyet_hourly,
    pevt_penman_kohler,
    pet_penman_hourly,
    pet_penman_monteith_hourly,
)

__all__ = [
    "wind_speed",
    "adjust_wind_height",
    "dewpoint_from_specific_humidity",
    "dewpoint_from_specific_humidity_cc",
    "dewpoint_august_roche_magnus",
    "kelvin_to_fahrenheit",
    "kelvin_to_celsius",
    "cloud_cover_davis",
    "thompson_sky_cover_radiation",
    "cloud_cover_thompson",
    "cloud_cover_linear",
    "hamon_weiss_wilson_clearsky",
    "hamon_weiss_wilson_actual_radiation",
    "pet_hargreaves",
    "pevt_penman_pyet_daily",
    "pevt_penman_pyet_hourly",
    "pevt_penman_kohler",
    "pet_penman_hourly",
    "pet_penman_monteith_hourly",
]