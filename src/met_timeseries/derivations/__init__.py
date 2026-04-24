"""
Source-agnostic derived variable calculations.
"""

from met_timeseries.derivations.wind import wind_speed, adjust_wind_height
from met_timeseries.derivations.humidity import (
    saturation_vapor_pressure_magnus,
    specific_humidity,
    relative_humidity,
    relative_humidity_from_specific_humidity,
    dewpoint_from_specific_humidity,
    dewpoint_from_specific_humidity_cc,
    dewpoint_august_roche_magnus,
)
from met_timeseries.derivations.temperature import (
    kelvin_to_fahrenheit,
    kelvin_to_celsius,
)
from met_timeseries.derivations.radiation import (
    net_radiation,
    clearsky_radiation,
    clearsky_array,
    cloud_cover_davis,
    sky_cover_radiation_thompson,
    cloud_cover_thompson,
    cloud_cover_linear,
    clearsky_radiation_hww,
    actual_radiation_hww,
)
from met_timeseries.derivations.pet import (
    pet_hargreaves,
    pet_penman_pyet_daily,
    pet_penman_pyet_hourly,
    pet_penman_kohler,
    pet_penman_hourly,
    pet_penman_monteith_hourly,
)

__all__ = [
    "wind_speed",
    "adjust_wind_height",
    "saturation_vapor_pressure_magnus",
    "specific_humidity",
    "relative_humidity",
    "relative_humidity_from_specific_humidity",
    "dewpoint_from_specific_humidity",
    "dewpoint_from_specific_humidity_cc",
    "dewpoint_august_roche_magnus",
    "kelvin_to_fahrenheit",
    "kelvin_to_celsius",
    "net_radiation",
    "clearsky_radiation",
    "clearsky_array",
    "cloud_cover_davis",
    "sky_cover_radiation_thompson",
    "cloud_cover_thompson",
    "cloud_cover_linear",
    "clearsky_radiation_hww",
    "actual_radiation_hww",
    "pet_hargreaves",
    "pet_penman_pyet_daily",
    "pet_penman_pyet_hourly",
    "pet_penman_kohler",
    "pet_penman_hourly",
    "pet_penman_monteith_hourly",
]
