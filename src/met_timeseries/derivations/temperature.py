from __future__ import annotations
import xarray as xr

def kelvin_to_fahrenheit(temp_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Fahrenheit."""
    return ((temp_k - 273.15) * 9/5 + 32).rename("temp_f")

def kelvin_to_celsius(temp_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius."""
    return (temp_k - 273.15).rename("temp_c")