from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pyet

from met_timeseries.derivations import constants, humidity, radiation

# --- FAO-56 Reference Crop (short grass, hourly Penman-Monteith) ---
FAO56_CN       = 37.0  # Numerator constant (kPa·s³/Mg/hr)
FAO56_CD_DAY   = 0.24  # Denominator constant, daytime
FAO56_CD_NIGHT = 0.96  # Denominator constant, nighttime

# Daily Methods
def hargreaves(daily_tmin, daily_tmax, lat: float):
    """Estimate daily PET using the Hargreaves (1985) method."""
    from mettoolbox.utils import radiation as met_rad

    tmean = (daily_tmin + daily_tmax) / 2.0
    trange = (daily_tmax - daily_tmin).clip(lower=0.0)

    ra_df = met_rad(tmean.to_frame(name="temp"), lat)
    ra = ra_df["ra"]

    with np.errstate(invalid="ignore"):
        pet = 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange)

    pet = pet.fillna(0.0).clip(lower=0.0)
    pet.name = "pet_hargreaves_mm"
    return pet

def penman_pan():
    raise NotImplementedError("Daily Penman Pan method is not yet implemented. Use pet_penman_kohler instead, which replicates the Kohler 1955 pan method and can be disaggregated to hourly resolution.")

def preistly_tayler():
    raise NotImplementedError()

def penman_knb(
    temp_mean_c: xr.DataArray,
    temp_min_c: xr.DataArray,
    temp_max_c: xr.DataArray,
    temp_dew_c: xr.DataArray,
    wind_speed_ms: xr.DataArray,
    solar_rad_mj: xr.DataArray  # Daily total solar radiation in MJ/m²/day
) -> xr.DataArray:
    """
    Computes daily pan evaporation replicating legacy Kohler-Nordenson-Baker (1959). 
    Refactored from 2017 TetraTech MetTool. This is also the same method that RESPEC uses

    Accepts metric xarray DataArrays, vectorizes the empirical English-unit 
    computations, and returns evaporation in metric mm/day.
    """
    # 1. Convert metric inputs to legacy English units
    #temp_min_f = temp_min_c * 1.8 + 32.0
    #temp_max_f = temp_max_c * 1.8 + 32.0
    #t_mean_f = (temp_min_f + temp_max_f) / 2.0
    t_mean_f = temp_mean_c*1.8 + 32.0
    t_dew_f = temp_dew_c * 1.8 + 32.0
    
    
    wind_mpd = wind_speed_ms * 53.6819       # 1 m/s = ~53.68 miles/day
    solar_langley = solar_rad_mj * 23.8845   # 1 MJ/m² = ~23.88 Langleys
    
    # Prevent log(0) errors using xarray/numpy clipping
    solar_langley = solar_langley.clip(min=25.0)
    
    # 3. Radiation Term (Equivalent to Qn * Delta)
    rad_exponent = (t_mean_f - 212.0) * (0.1024 - 0.01066 * np.log(solar_langley))
    rad_term = np.exp(rad_exponent) - 0.0001
    
    #4. Vapor Pressure Deficit (es - ea)
    # Current: Lamoreux on mean temp (English units, stays in inHg)
    es = humidity.vapor_pressure_lamoreux(t_mean_f)

    # Option B (commented): Lamoreux averaged over min/max — KNB (1959) original
    #es = (humidity.vapor_pressure_lamoreux(temp_max_f) + humidity.vapor_pressure_lamoreux(temp_min_f)) / 2.0 

    # Option C (commented): Magnus on mean temp — metric, modern
    #es = humidity.vapor_pressure_magnus(temp_mean_c) / 3.38639 # convert kPa to inHg

    # Option D (commented): Magnus averaged over min/max — metric, WMO standard
    #es = (humidity.vapor_pressure_magnus(temp_max_c) + humidity.vapor_pressure_magnus(temp_min_c)) / 2.0 / 3.38639# convert kPa to inHg

    ea = humidity.vapor_pressure_lamoreux(t_dew_f)
    #ea = humidity.vapor_pressure_magnus(temp_dew_c) / 3.38639

    vpd = (es - ea).clip(min=0.0)
    
    # 5. Aerodynamic Term (using the PAN psychrometric constant: 0.025) (Equivalent to Ea * gamma) 
    gamma_pan = 0.0105 
    #gamma_pan = 0.025
    aero_term = gamma_pan * (vpd ** 0.88) * (0.37 + 0.0041 * wind_mpd)
    #aero_term = 0.0105 * (vpd ** 0.88) * (0.37 + 0.0041 * wind_mpd)

    # 6. Slope of saturation vapor pressure curve (Delta)
    # Current: legacy KNB empirical formula (English units, internally consistent)
    delta = (47987800000.0 * np.exp(-7482.6 / (t_mean_f + 398.36))) / ((t_mean_f + 398.36) ** 2)

    # Alternate: modern humidity module
    #delta = humidity.delta_svp(temp_mean_c) #kPa/°C convert to 

    # 7. Compute Pan Evaporation in Inches
    pan_evap_in = (rad_term + aero_term) / (delta + gamma_pan)

    # 8. Convert back to Metric (mm/day) and clip negative values
    pan_evap_mm = (pan_evap_in * 25.4).clip(min=0.0)
    
    return pan_evap_mm.rename("penman_knb").assign_attrs({"units": "mm/day"})    


def oudin(
    temperature_c: xr.DataArray,
) -> xr.DataArray:
    """
    Computes daily Potential Evapotranspiration (mm/day) using the Oudin (2005) method.
    
    This method is highly optimized for rainfall-runoff models and relies purely 
    on mean daily temperature and extraterrestrial radiation.

    Args:
        temperature_c: Mean daily air temperature in °C.
        ra_mj: Extraterrestrial radiation (Ra) at the top of the atmosphere in MJ/m²/day.
               (Can be calculated purely from latitude and day of the year).
               
    Returns:
        xr.DataArray: Daily PET in mm/day.
    """
    ra_mj = radiation.extra_radiation(temperature_c) #Note it just takes the temperature to get the time coordinate and lat coordinate, the actual value is not used in the calculation
    # 1. Convert Ra (Energy) into mm/day (Water Equivalent)
    # Using 2.45 MJ/kg as the standard latent heat of vaporization
    ra_mm_day = ra_mj / 2.45

    # 2. Apply the Oudin mathematical formula
    # Extraterrestrial radiation scaled by a simple temperature index
    oudin_pet = (ra_mm_day / 100.0) * (temperature_c + 5.0)

    # 3. Apply the threshold constraint
    # Oudin explicitly defined PET as 0 for any day where mean temp drops below -5°C
    #pet_daily = xr.where(temperature_c > -5.0, oudin_pet, 0.0)

    # 4. Standard safety clip to prevent physically impossible negative evaporation
    pet_daily = oudin_pet.clip(min=0.0)

    return pet_daily.rename("pet_oudin").assign_attrs(
        {"units": "mm/day", "long_name": "Oudin Potential Evapotranspiration"}
    )


# Hourly Methods


def penman_monteith_asce(
    shortwave_down: xr.DataArray,
    longwave_down: xr.DataArray,
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
    wind_speed: xr.DataArray,
    pressure: xr.DataArray,
    albedo: float = 0.23, # 0.23 is standard for PM reference crop
    cn: float = 37.0,     # Short crop (grass) default
    cd_day: float = 0.24, # Short crop day default
    cd_night: float = 0.96 # Short crop night default
) -> xr.DataArray:
    """Compute hourly ASCE Standardized Penman-Monteith Reference ET (mm/hour)."""

    LAMBDA = 2.45 #MJ/kg
    PSYCHROMETRIC_COEFFICIENT = 0.000665 #kPa/degC
    
    # 1. Net Radiation (Hourly Volume)
    net_radiation = radiation.net_radiation(shortwave_down, longwave_down, temperature, albedo)
    rn_mj = net_radiation * 0.0036  

    # 2. Dynamic Hourly Soil Heat Flux (G) and Aerodynamic Coefficient (Cd)
    # ASCE defines "daytime" as any hour where Rn > 0
    # Current: ASCE-standard Rn > 0 proxy
    is_daytime = rn_mj > 0

    # Alternate: pvlib solar elevation (used in all other methods)
    # daytime_mask = radiation.daytime_mask_solar_elevation(shortwave_down, min_solar_elevation=10.0)


    g_mj = xr.where(is_daytime, 0.10 * rn_mj, 0.50 * rn_mj)
    cd_dynamic = xr.where(is_daytime, cd_day, cd_night)

    # 3. Available Energy
    available_energy = rn_mj - g_mj

    # 4. Vapor Pressures
    e_s = humidity.vapor_pressure_magnus(temperature)
    e_a = humidity.vapor_pressure_magnus(dewpoint)
    vapor_pressure_deficit = (e_s - e_a).clip(min=0.0)

    # 5. Psychrometric & Thermodynamics
    # Current: simplified constant × pressure
    pressure_kpa = pressure / 1000.0
    gamma = PSYCHROMETRIC_COEFFICIENT * pressure_kpa   # fixed λ assumed

    # Alternate: temperature-dependent λ (now available in humidity module)
    # lambda_v = humidity.latent_heat_linear(temperature)
    # gamma = humidity.psychrometric_constant_dynamic(pressure_kpa, lambda_v)

    delta = humidity.delta_svp(temperature, humidity.VAPOR_B_MAGNUS)

    # 6. ASCE PM Equation
    # 0.408 is the RECIPROCAL_LAMBDA_20C
    numerator = (
        humidity.RECIPROCAL_LAMBDA_20C * delta * available_energy 
        + gamma * (cn / (temperature + 273.15)) * wind_speed * vapor_pressure_deficit
    )
    
    denominator = delta + gamma * (1 + cd_dynamic * wind_speed)

    pet_hourly = numerator / denominator  
    
    return xr.DataArray(
        pet_hourly.clip(min=0.0),  
        dims=shortwave_down.dims,
        coords=shortwave_down.coords,
        name="pet_penman_monteith_hourly_mm",
    )


def _disaggregate_pet_trapezoidal(
    daily_pet: xr.DataArray,
) -> xr.DataArray:
    """
    Distributes daily PET into an hourly curve using a legacy trapezoidal 
    solar day-length approximation. Refactored from 2017 TetraTech MetTool.
    
    Args:
        daily_pet: DataArray containing daily PET values.
           """
    # 1. Extract coordinates and time arrays
    hourly_template = daily_pet.resample(time="1h").ffill()
    lat = hourly_template.coords["lat"]
    lat_rad = np.radians(lat)
    
    # Use exact datetime properties instead of the legacy '30.5 * Month' approximation
    doy = hourly_template.coords["time"].dt.dayofyear
    hour = hourly_template.coords["time"].dt.hour
    
    # 2. Solar Geometry (Preserving legacy empirical constants for exact replication)
    declination = 0.40928 * np.cos(0.0172141 * (172.0 - doy))
    
    # SS / CS mathematically reduces to -tan(lat) * tan(declination)
    x2 = -np.tan(lat_rad) * np.tan(declination)
    
    # Safety clip to prevent domain errors in polar regions during solstices
    x2 = x2.clip(min=-1.0, max=1.0)
    
    # Replace the legacy atan(x/sqrt(1-x^2)) hack with native arccos
    day_length = 7.6394 * np.arccos(x2)
    sunrise = 12.0 - day_length / 2.0
    
    # 3. Prevent Division by Zero (Polar Night)
    # If day_length is 0, we temporarily set it to 1.0 to let the math evaluate safely.
    # The valid_day mask ensures the final output is still 0.0 for those days.
    valid_day = day_length > 0.0
    safe_day_length = day_length.where(valid_day, 1.0)
    
    # 4. Trapezoid Parameters
    dtr2 = safe_day_length / 2.0
    dtr4 = safe_day_length / 4.0
    
    crad = (2.0 / 3.0) / dtr2    # Peak multiplier
    sl = crad / dtr4             # Slope of the curve
    
    tr2 = sunrise + dtr4
    tr3 = tr2 + dtr2
    tr4 = sunrise + safe_day_length # Sunset
    
    # 5. Build the Piecewise Hourly Mask using xr.where
    # Condition 1: Morning Ramp Up
    fraction = xr.where(
        valid_day & (hour > sunrise) & (hour <= tr2),
        (hour - sunrise) * sl,
        0.0
    )
    
    # Condition 2: Midday Peak (Flat Top)
    fraction = xr.where(
        valid_day & (hour > tr2) & (hour <= tr3),
        crad,
        fraction
    )
    
    # Condition 3: Afternoon Ramp Down
    fraction = xr.where(
        valid_day & (hour > tr3) & (hour <= tr4),
        crad - (hour - tr3) * sl,
        fraction
    )
    
    # 6. Apply to Daily Data
    # Forward-fill the daily PET values onto the hourly grid shape
    daily_pet_hourly_grid = daily_pet.resample(time="1h").ffill().reindex_like(hourly_template, method="ffill")
    
    hourly_pet = daily_pet_hourly_grid * fraction
    
    return hourly_pet.rename("pet_hourly_trapezoidal")

def _disaggregate_daily_to_hourly_solar(
    daily: xr.DataArray,
    shortwave_hourly: xr.DataArray,
    daytime_mask: xr.DataArray,
) -> xr.DataArray:
    """
    Disaggregate a daily value to hourly resolution using solar radiation
    as a temporal weight.

    Signal flow:
        daily total  ×  (hourly SW / daily SW sum)  →  hourly value

    PET is allocated only during daytime hours (where daytime_mask is True).
    Nighttime hours and days with no daytime SW receive 0.

    Args:
        daily:            Daily values to disaggregate (any units per day).
        shortwave_hourly: Hourly incoming shortwave radiation (W/m²),
                          used as the weighting signal.
        daytime_mask:     Boolean mask marking daytime hours, typically from
                          clearsky_radiation_ineichen(...) with a minimum solar elevation threshold.

    Returns:
        Hourly DataArray with the same dims/coords as shortwave_hourly.
    """
    # Zero out nighttime SW so only daytime hours receive weight
    sw_daytime = shortwave_hourly.where(daytime_mask, 0.0)

    # Daily sum of daytime SW — NaN where no daytime SW exists (polar night etc.)
    sw_daily_sum = sw_daytime.resample(time="1D").sum()
    sw_daily_sum = sw_daily_sum.where(sw_daily_sum > 0.0, other=np.nan)

    # Broadcast daily values and SW sum back to hourly resolution
    daily_bcast   = daily.reindex_like(shortwave_hourly, method="ffill")
    sw_sum_bcast  = sw_daily_sum.reindex_like(shortwave_hourly, method="ffill")

    return (daily_bcast * sw_daytime / sw_sum_bcast).fillna(0.0).clip(min=0.0)

