from __future__ import annotations
import numpy as np
import xarray as xr

from met_timeseries.derivations import constants
# --- Vapor Pressure (Magnus formula family) ---
SAT_VP_0C_KPA = 0.6108          # Saturation VP at 0 °C, FAO-56 base (kPa)
SAT_VP_0C_ARM = 0.6112          # Saturation VP at 0 °C, ARM base (kPa)
VAPOR_A_MAGNUS = 17.27          # Magnus coefficient A (dimensionless)
VAPOR_B_MAGNUS = 237.3          # Magnus coefficient B, FAO-56 (°C)
VAPOR_B_TETENS = 237.7          # Magnus coefficient B, August-Roche-Magnus (°C)

# --- Clausius-Clapeyron ---
def delta_svp(
    temperature: xr.DataArray,
    b: float = constants.VAPOR_B_MAGNUS,
) -> xr.DataArray:
    """
    Slope of the saturation vapor pressure curve (Δ) in kPa/°C.

    Analytical derivative of the Magnus formula:
        Δ = (A · B · eₛ) / (T + B)²
          = 4098 · eₛ / (T + 237.3)²    ← FAO-56 form

    Signal flow:
        T  →  eₛ (vapor_pressure_magnus)  →  Δ

    Args:
        temperature: Air temperature (°C).
        b:           Magnus B coefficient (°C). Defaults to FAO-56 (237.3).
                     Pass constants.VAPOR_B_TETENS (237.7) for ARM variant.
    """
    es = vapor_pressure_magnus(temperature, b=b)
    return (constants.VAPOR_A_MAGNUS * b * es) / (temperature + b) ** 2

def vapor_pressure_magnus(
    temperature: xr.DataArray,
    a: float = constants.VAPOR_A_MAGNUS,
    b: float = constants.VAPOR_B_MAGNUS,
    base: float = 0.6108,
) -> xr.DataArray:
    """
    Calculate vapor pressure (e) in kPa using the Magnus formula.

    Pass air temperature to get saturation vapor pressure (eₛ).
    Pass dewpoint temperature to get actual vapor pressure (eₐ).

    Args:
        temperature: Air or dewpoint temperature (°C).
        a:    Magnus coefficient A (default 17.27, FAO-56).
        b:    Magnus coefficient B in °C (default 237.3, FAO-56).
              Pass constants.VAPOR_B_TETENS (237.7) for August-Roche-Magnus variant.
        base: Vapor pressure at 0 °C in kPa (default 0.6108, FAO-56).
              Pass 0.6112 for August-Roche-Magnus variant.
    """
    return base * np.exp((a * temperature) / (temperature + b))

def vapor_pressure_mixing(
    specific_humidity: xr.DataArray,  # kg/kg
    pressure: xr.DataArray,           # Pa
) -> xr.DataArray:
    """
    Derive actual vapor pressure (kPa) from specific humidity and pressure
    via mixing ratio.

    Signal flow: q  →  w (mixing ratio)  →  e_a (kPa)

    Args:
        specific_humidity: Specific humidity (kg/kg)
        pressure:          Atmospheric pressure (Pa)
    """
    w = specific_humidity / (1.0 - specific_humidity)
    return (w / (constants.EPSILON + w)) * (pressure / 1000.0)

def dewpoint_magnus(
    vapor_pressure: xr.DataArray,
    a: float = constants.VAPOR_A_MAGNUS,
    b: float = constants.VAPOR_B_TETENS,
    base: float = SAT_VP_0C_ARM,
) -> xr.DataArray:
    """
    Invert the Magnus formula to recover dewpoint temperature (°C) from
    vapor pressure (kPa).

    Algebraic inverse of vapor_pressure_magnus:
        e = base · exp(A·T / (T + B))  →  Td = B · ln(e/base) / (A - ln(e/base))

    Defaults to ARM constants (VAPOR_B_TETENS, 0.6112) since this inversion is
    most commonly used in the August-Roche-Magnus dewpoint pathway. Pass
    VAPOR_B_MAGNUS / 0.6108 for the FAO-56 variant.

    Args:
        vapor_pressure: Actual vapor pressure (kPa).
        a:    Magnus coefficient A (default 17.27).
        b:    Magnus coefficient B in °C (default 237.7, ARM).
        base: Vapor pressure at 0 °C in kPa (default 0.6112, ARM).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_e = np.log(vapor_pressure / base)
        dewpoint = (b * ln_e) / (a - ln_e)
    return dewpoint.rename("dewpoint_c")


def dewpoint_cc(
    vapor_pressure: xr.DataArray,
    ref_vp: float = 0.6113,
    cc_const: float = 0.0001844,
) -> xr.DataArray:
    """
    Invert vapor pressure (kPa) to dewpoint temperature (°C) via the
    linearised Clausius-Clapeyron equation.

        Td = 1 / (1/T₀ - (1/L) · ln(e/e₀))  - 273.15

    where T₀ = 273.15 K, e₀ = ref_vp kPa, and cc_const approximates 1/L.

    Args:
        vapor_pressure: Actual vapor pressure (kPa).
        ref_vp:    Reference vapor pressure at 0 °C (kPa). Default 0.6113.
        cc_const:  Linearised latent heat coefficient (K⁻¹). Default 0.0001844.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        td_k = 1.0 / (1.0 / 273.15 - cc_const * np.log(vapor_pressure / ref_vp))
    return (td_k - 273.15).rename("dewpoint_c")


def specific_humidity(
    dewpoint: xr.DataArray, # Celsius
    pressure: xr.DataArray,  # Pa
) -> xr.DataArray:
    """Calculate specific humidity (q) in kg/kg.
    
    Args:
        pressure: Atmospheric pressure in Pa
    """
    e_a = vapor_pressure_magnus(dewpoint)   # kPa
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

    e_s = vapor_pressure_magnus(temperature)  # kPa
    e = (specific_humidity * pressure_kpa) / (constants.EPSILON + specific_humidity)
    rh = (e / e_s) * 100.0

    return xr.DataArray(
        rh.clip(0.0, 100.0),
        dims=temperature.dims,
        coords=temperature.coords,
        name="relative_humidity",
    )

def relative_humidity(
    temperature: xr.DataArray, # Celsius
    dewpoint: xr.DataArray, # Celsius
    pressure: xr.DataArray, # Pa
) -> xr.DataArray:
    """Compute relative humidity from specific humidity, pressure, and temperature (°C)."""
    sh = specific_humidity(dewpoint, pressure)
    rh = relative_humidity_from_specific_humidity(temperature,sh,pressure)
  
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
    e_a = vapor_pressure_mixing(specific_humidity, pressure)
    dewpoint = dewpoint_cc(e_a)   # or dewpoint_cc(e_a)
    return xr.DataArray(
        dewpoint,
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
    
    e_a = vapor_pressure_mixing(specific_humidity, pressure)
    dewpoint = dewpoint_magnus(e_a)  

    return xr.DataArray(
        dewpoint,
        dims=temperature.dims,
        coords=temperature.coords,
        name="dewpoint_c",
    )
