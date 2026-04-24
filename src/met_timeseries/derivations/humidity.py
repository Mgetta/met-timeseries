from __future__ import annotations
import numpy as np
import xarray as xr

from met_timeseries.derivations import constants


def saturation_vapor_pressure_magnus(temperature: xr.DataArray) -> xr.DataArray:
    """
    Calculate saturation vapor pressure (eₛ) in kPa using the Magnus formula.
    Assumes temperature is in Celsius.
    """
    return 0.6108 * np.exp(
        constants.VAPOR_A_MAGNUS * temperature / (temperature + constants.VAPOR_B_MAGNUS)
    )


def specific_humidity(
    dewpoint: xr.DataArray, # Celsius
    pressure: xr.DataArray,  # Pa
) -> xr.DataArray:
    """Calculate specific humidity (q) in kg/kg.
    
    Args:
        pressure: Atmospheric pressure in Pa
    """
    e_a = saturation_vapor_pressure_magnus(dewpoint)   # kPa
    pressure_kpa = pressure / 1000.0                   # Pa → kPa

    q = (constants.EPSILON * e_a) / (pressure_kpa - (1 - constants.EPSILON) * e_a)
    return q.rename("specific_humidity")

def relative_humidity_from_specific_humidity(
    temperature: xr.DataArray,  # Celsius
    specific_humidity: xr.DataArray,  # kg/kg
    pressure: xr.DataArray,  # Pa
) -> xr.DataArray:
    """Compute relative humidity from specific humidity, pressure, and temperature (°C)."""
    pressure_kpa = pressure / 1000.0

    e_s = saturation_vapor_pressure_magnus(temperature)  # kPa
    e = (specific_humidity * pressure_kpa) / (constants.EPSILON + specific_humidity)
    rh = (e / e_s) * 100.0

    return xr.DataArray(
        rh.clip(0.0, 100.0),
        dims=temperature.dims,
        coords=temperature.coords,
        name="relative_humidity",
    )

def relative_humidity(
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Compute relative humidity from specific humidity, pressure, and temperature (°C)."""
    pressure_hpa = pressure / 100.0

    sh = specific_humidity(dewpoint, pressure)

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

def dewpoint_from_specific_humidity(
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute dewpoint temperature from specific humidity using MetPy."""
    from metpy.calc import dewpoint_from_specific_humidity as _mpc_dp
    from metpy.units import units

    if pressure is None:
        pres_pa = np.full_like(specific_humidity.values, 101325.0) * units("Pa")
    else:
        pres_pa = pressure.values * units("Pa")

    spfh = specific_humidity.values * units("kg/kg")
    dp = _mpc_dp(pres_pa, spfh)

    return xr.DataArray(
        dp.to("degC").magnitude,
        dims=specific_humidity.dims,
        coords=specific_humidity.coords,
        name="dewpoint_c",
    )

def dewpoint_from_specific_humidity_cc(
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Compute dewpoint from specific humidity using Clausius-Clapeyron."""
    q = specific_humidity.values.astype(float)
    p_kpa = pressure.values.astype(float) / 1000.0  

    w = q / (1.0 - q)
    e_a = (w / (0.622 + w)) * p_kpa

    REF_VP = 0.6113  
    CC_CONST = 0.0001844  

    with np.errstate(divide="ignore", invalid="ignore"):
        td_k = 1.0 / (1.0 / 273.15 - CC_CONST * np.log(e_a / REF_VP))

    td_c = td_k - 273.15

    return xr.DataArray(
        td_c,
        dims=specific_humidity.dims,
        coords=specific_humidity.coords,
        name="dewpoint_c",
    )

def dewpoint_august_roche_magnus(
    temperature: xr.DataArray,
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray,
) -> xr.DataArray:
    """Compute dewpoint temperature using the August-Roche-Magnus equation."""
    relative_humidity = relative_humidity_from_specific_humidity(
        temperature, specific_humidity, pressure
    )

    e_s = 6.112 * np.exp(
        (constants.VAPOR_A_MAGNUS * temperature) / (constants.VAPOR_B_TETENS + temperature)
    )
    e = (relative_humidity / 100) * e_s

    with np.errstate(divide="ignore", invalid="ignore"):
        dewpoint = (constants.VAPOR_B_TETENS * np.log(e / 6.112)) / (
            constants.VAPOR_A_MAGNUS - np.log(e / 6.112)
        )

    return xr.DataArray(
        dewpoint,
        dims=temperature.dims,
        coords=temperature.coords,
        name="dewpoint_c",
    )