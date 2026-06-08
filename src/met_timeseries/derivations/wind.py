from __future__ import annotations
import numpy as np
import xarray as xr
 
# Aeordynamics and Kinetics functions
def adjust_wind_height(
    wind_speed: xr.DataArray,
    z_from: float = 10.0,
    z_to: float = 2.0,
    method: str = "logarithmic",
    alpha: float = 1 / 7,
) -> xr.DataArray:
    """Adjust wind speed between measurement heights."""
    # Local coefficients for FAO-56 Eq 47 logarithmic profile assumption (short grass)


    if method == "logarithmic":
        FAO_MULTIPLIER = 67.8
        FAO_OFFSET = 5.42
        factor = np.log(FAO_MULTIPLIER * z_to - FAO_OFFSET) / np.log(FAO_MULTIPLIER * z_from - FAO_OFFSET)
        adjusted = wind_speed * factor
    elif method == "power_law":
        adjusted = wind_speed * (z_to / z_from) ** alpha
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'logarithmic' or 'power_law'.")

    return adjusted.rename("wind_speed_ms")

def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Compute wind speed from U and V components."""

    return ((u**2 + v**2)**0.5).rename("wind_speed_ms")

# Transfer functions
def wind_function_kohler(wind_speed_ms: xr.DataArray) -> xr.DataArray:
    """
    Kohler et al. (1955) empirical wind function for pan evaporation.

    f(u) = a + b·u   where u is wind speed in km/hr

    Signal flow:
        u (m/s)  →  u (km/hr)  →  f(u) (mm/day/kPa)

    Args:
        wind_speed_ms: Wind speed at 2m (m/s)
    """
    KOHLER_WIND_A = 0.005    # Intercept
    KOHLER_WIND_B = 0.00085  # Slope (per km/hr)

    wind_kmhr = wind_speed_ms * 3.6
    return KOHLER_WIND_A + KOHLER_WIND_B * wind_kmhr


def wind_function_fao56(
    wind_speed: xr.DataArray,
    temperature: xr.DataArray,
    cn: float,
) -> xr.DataArray:
    """
    FAO-56 aerodynamic wind term for the Penman-Monteith numerator.

    f(u) = cn / (T + 273) · u

    Signal flow:
        u, T  →  aerodynamic transfer coefficient (mm/hr/kPa per kPa/°C)

    Args:
        wind_speed:  Wind speed at 2m (m/s)
        temperature: Air temperature (°C)
        cn:          FAO-56 numerator constant (default 37.0, short reference grass)
    """
    return (cn / (temperature + 273.0)) * wind_speed