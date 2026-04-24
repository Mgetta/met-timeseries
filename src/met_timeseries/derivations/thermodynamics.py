from __future__ import annotations

import numpy as np
import xarray as xr

from met_timeseries.derivations import constants


def calculate_net_radiation(
    shortwave_down: xr.DataArray,
    longwave_down: xr.DataArray,
    temperature: xr.DataArray,
    albedo: float = 0.23,
    emissivity: float = 0.97,
) -> xr.DataArray:
    """Calculate net radiation (R_n) in W/m²."""
    shortwave_up = shortwave_down * albedo
    longwave_up = emissivity * constants.STEFAN_BOLTZMANN * temperature**4
    net_radiation = (shortwave_down - shortwave_up) + (longwave_down - longwave_up)
    return net_radiation.rename("net_radiation")


def svp_magnus(temperature: xr.DataArray) -> xr.DataArray:
    """
    Calculate saturation vapor pressure (eₛ) in kPa using the Magnus formula.
    Assumes temperature is in Celsius.
    """
    return 0.6108 * np.exp(
        constants.VAPOR_A_MAGNUS * temperature / (temperature + constants.VAPOR_B_MAGNUS)
    )


def calculate_specific_humidity(
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Calculate specific humidity (q) in kg/kg."""
    e_a = svp_magnus(dewpoint)
    q = (constants.EPSILON * e_a) / (pressure - (1 - constants.EPSILON) * e_a)
    return q.rename("specific_humidity")


def calculate_relative_humidity(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Compute relative humidity from specific humidity, pressure, and temperature (°C)."""
    pressure_hpa = pressure / 100.0

    e_s = 6.112 * np.exp(
        (constants.VAPOR_A_MAGNUS * temperature) / (constants.VAPOR_B_TETENS + temperature)
    )
    e = (specific_humidity * pressure_hpa) / (constants.EPSILON + specific_humidity)
    rh = (e / e_s) * 100.0

    return xr.DataArray(
        rh.clip(0.0, 100.0),
        dims=temperature.dims,
        coords=temperature.coords,
        name="relative_humidity",
    )