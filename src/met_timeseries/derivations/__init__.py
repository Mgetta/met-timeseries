"""
Source-agnostic derived variable calculations.
"""

from met_timeseries.derivations.wind import wind_speed, adjust_wind_height
from met_timeseries.derivations.humidity import (
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
    clearsky_radiation_geometric,
    clearsky_radiation_ineichen,
    daytime_mask_solar_elevation,
    clearsky_radiation_hww,
    cloud_cover_davis,
    sky_cover_radiation_thompson,
    cloud_cover_thompson,
    cloud_cover_linear,
    actual_radiation_hww,
)
from met_timeseries.derivations.pet import (
    pet_hargreaves,
    pet_penman_kohler,
    pet_penman_knb,
    pet_oudin,
    penman_monteith_asce,
    _disaggregate_daily_to_hourly_solar,
    _disaggregate_pet_trapezoidal,
)
