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

def pet_hargreaves(daily_tmin, daily_tmax, lat: float):
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

def pet_penman_pyet_daily(
    shortwave_hourly: xr.DataArray,
    temperature_hourly: xr.DataArray,
    dewpoint_hourly: xr.DataArray,
    wind_hourly: xr.DataArray,
    elevation: "xr.DataArray | float" = 0.0,
    albedo: float = 0.06,
) -> xr.DataArray:
    """Daily Penman pan evaporation on a spatial grid via pyet (mm/day).

    Note:
        shortwave_hourly is assumed to be at 1-hour intervals. The factor 0.0036
        converts W/m² to MJ/m²/hr (= W/m² × 3600 s/hr × 1e-6 MJ/J).
        Sub-hourly or super-hourly data will give incorrect daily totals.
    """
    rs_daily    = (shortwave_hourly * 0.0036).resample(time="1D").sum()  
    tmean_daily = temperature_hourly.resample(time="1D").mean()          
    wind_daily  = wind_hourly.resample(time="1D").mean()                 
    ea_daily    =  humidity.vapor_pressure_magnus(dewpoint_hourly).resample(time="1D").mean()                                          

    lat_rad = np.radians(shortwave_hourly.coords["lat"])

    pet = pyet.penman(
        tmean=tmean_daily,
        wind=wind_daily,
        rs=rs_daily,
        ea=ea_daily,
        lat=lat_rad,
        elevation=elevation,
        albedo=albedo,
    )

    return pet.clip(min=0.0).rename("pet_penman_mm_day").assign_attrs(
        {"units": "mm/day", "albedo": albedo}
    )

def pet_penman_pyet_hourly(
    shortwave_hourly: xr.DataArray,
    temperature_hourly: xr.DataArray,
    dewpoint_hourly: xr.DataArray,
    wind_hourly: xr.DataArray,
    elevation: "xr.DataArray | float" = 0.0,
    albedo: float = 0.06,
    min_solar_elevation: float = 10.0,
) -> xr.DataArray:
    """Hourly Penman pan evaporation on a spatial grid (mm/hr)."""
    pet_daily = pet_penman_pyet_daily(
        shortwave_hourly, temperature_hourly, dewpoint_hourly,
        wind_hourly, elevation, albedo,
    )

    daytime_mask = radiation.daytime_mask_solar_elevation(
        shortwave_hourly, min_solar_elevation=min_solar_elevation,
    )

    pet_hourly = _disaggregate_daily_to_hourly_solar(
        pet_daily, shortwave_hourly, daytime_mask
    )

    return pet_hourly.rename("pet_penman_mm_hr").assign_attrs(
        {"units": "mm/hr", "albedo": albedo}
    )

def pet_penman_kohler(
    shortwave_hourly: xr.DataArray,
    temperature_hourly: xr.DataArray,
    dewpoint_hourly: xr.DataArray,
    wind_hourly: xr.DataArray,
    pressure: xr.DataArray | float = 101325.0,
    albedo: float = 0.06,
    min_solar_elevation: float = 10.0,
) -> xr.DataArray:
    """Hourly Penman pan evaporation via Kohler et al. (1955) (mm/hr).

    Note:
        shortwave_hourly is assumed to be at 1-hour intervals. The factor 0.0036
        converts W/m² to MJ/m²/hr (= W/m² × 3600 s/hr × 1e-6 MJ/J).
        Sub-hourly or super-hourly data will give incorrect daily totals.

    Args:
        pressure: Atmospheric pressure in Pa (default 101325.0 Pa = standard atmosphere).
    """
    rs_daily    = (shortwave_hourly * 0.0036).resample(time="1D").sum() 
    tmean_daily = temperature_hourly.resample(time="1D").mean()          
    dp_daily    = dewpoint_hourly.resample(time="1D").mean()             
    wind_daily  = wind_hourly.resample(time="1D").mean()                 

    es = humidity.vapor_pressure_magnus(tmean_daily)
    ea = humidity.vapor_pressure_magnus(dp_daily)
    
    vpd      = (es - ea).clip(min=0.0)
    delta = humidity.delta_svp(tmean_daily, humidity.VAPOR_B_MAGNUS)
    pressure_kpa = pressure / 1000.0
    gamma    = constants.PSYCHROMETRIC_COEFFICIENT * pressure_kpa                                        
    lambda_v = constants.LAMBDA_0 - constants.LAMBDA_T * tmean_daily                                
    rn       = rs_daily * (1.0 - albedo)                                      
    f_u      = 0.005 + 0.00085 * (wind_daily * 3.6)                          
    ea_term  = f_u * vpd


    pet_daily = ((delta * (rn / lambda_v) + gamma * ea_term) / (delta + gamma)).clip(min=0.0)

    daytime_mask = radiation.daytime_mask_solar_elevation(
        shortwave_hourly, min_solar_elevation=min_solar_elevation,
    )

    pet_hourly = _disaggregate_daily_to_hourly_solar(
        pet_daily, shortwave_hourly, daytime_mask
    )
    # sw_daytime   = shortwave_hourly.where(daytime_mask, 0.0)
    # sw_daily_sum = sw_daytime.resample(time="1D").sum()
    # sw_daily_sum = sw_daily_sum.where(sw_daily_sum > 0.0, other=np.nan)

    # pet_broadcast = pet_daily.reindex_like(shortwave_hourly, method="ffill")
    # sw_sum_bcast  = sw_daily_sum.reindex_like(shortwave_hourly, method="ffill")

    # pet_hourly = (pet_broadcast * sw_daytime / sw_sum_bcast).fillna(0.0).clip(min=0.0)

    return pet_hourly.rename("pet_penman_kohler_mm_hr").assign_attrs(
        {"units": "mm/hr", "albedo": albedo}
    )

def pet_penman_hourly(
    shortwave_down: xr.DataArray,
    longwave_down: xr.DataArray,
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
    wind_speed: xr.DataArray,
    pressure: xr.DataArray,
    albedo: float = 0.23,
) -> xr.DataArray:
    """Compute hourly Penman Pan Evaporation (mm/hour).

    Args:
        temperature: Air temperature in °C.
        dewpoint: Dewpoint temperature in °C.
        pressure: Atmospheric pressure in Pa.
    """
    cn = FAO56_CN
    cd = FAO56_CD_DAY

    net_radiation = radiation.net_radiation(shortwave_down, longwave_down, temperature, albedo)
    net_radiation_mj = net_radiation * 0.0036  

    e_s = humidity.vapor_pressure_magnus(temperature)
    e_a = humidity.vapor_pressure_magnus(dewpoint)

    pressure_kpa = pressure / 1000.0
    gamma = constants.PSYCHROMETRIC_COEFFICIENT * pressure_kpa  

    delta = humidity.delta_svp(temperature, humidity.VAPOR_B_MAGNUS)
    vapor_pressure_deficit = e_s - e_a

    numerator = constants.RECIPROCAL_LAMBDA_20C * delta * net_radiation_mj + gamma * (cn / (temperature + 273.0)) * wind_speed * vapor_pressure_deficit
    denominator = delta + gamma * (1 + cd * wind_speed)

    pet_hourly = numerator / denominator  
    pet_hourly = xr.DataArray(
        pet_hourly.clip(0.0),  
        dims=shortwave_down.dims,
        coords=shortwave_down.coords,
        name="pet_penman_hourly_mm",
    )
    return pet_hourly

def pet_penman_monteith_hourly(
    temperature: xr.DataArray,
    wind_speed: xr.DataArray,
    shortwave: xr.DataArray,
    dewpoint: xr.DataArray,
    pressure: xr.DataArray,
    albedo = 0.23
) -> xr.DataArray:
    """Compute hourly FAO-56 Penman-Monteith reference ET (mm/hour).

    Args:
        temperature: Air temperature in °C.
        dewpoint: Dewpoint temperature in °C.
        shortwave: Incoming shortwave radiation in W/m².
        pressure: Surface pressure in kPa.
    """
    cn = FAO56_CN
    cd_day = FAO56_CD_DAY
    cd_night = FAO56_CD_NIGHT

    # Compute clear-sky radiation for FAO-56 Eq 39 cloudiness correction (Rs/Rso)
    rso_xr = radiation.clearsky_radiation_ineichen(shortwave)
    
    t = temperature.values.astype(float)
    u2 = wind_speed.values.astype(float)
    rs_wm2 = shortwave.values.astype(float)
    td = dewpoint.values.astype(float)

    rs = rs_wm2 * 0.0036  # W/m² → MJ/m²/hr
    #pressure = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26  
    gamma = constants.PSYCHROMETRIC_COEFFICIENT * pressure / 10


    es = humidity.vapor_pressure_magnus(t)
    ea = humidity.vapor_pressure_magnus(td)

    delta = humidity.delta_svp(t, humidity.VAPOR_B_MAGNUS)
    rns = (1.0 - albedo) * rs
    
    # # FAO-56 Eq 39: cloudiness correction (1.35 * Rs/Rso - 0.35).
    # # When rso = 0 (nighttime) the ratio is undefined; use 1.0 (clear-sky
    # # assumption) so that rnl remains physically bounded.
    # # The cloudiness factor is clipped to [0.05, 1.0] to prevent physically
    # # invalid negative net longwave values when Rs/Rso < 0.259.
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     rs_over_rso = np.clip(np.where(rso > 0, rs / rso, 1.0), 0.0, 1.0)
    # cloudiness_factor = np.clip(1.35 * rs_over_rso - 0.35, 0.05, 1.0)
    # rnl = sigma_h * (t_k**4) * (humid_a - humid_b * np.sqrt(np.maximum(ea, 0.0))) * cloudiness_factor
    rnl = radiation.net_longwave_brutsaert(t, ea, 
                                        clearsky_shortwave=rso_xr,  # clearsky
                                        shortwave_down=shortwave)    # observed

    rnl = rnl* 0.0036  # W/m² → MJ/m²/hr
    rn = rns - rnl
    is_daytime = rn > 0.0
    g = np.where(is_daytime, 0.1 * rn, 0.5 * rn)
    cd = np.where(is_daytime, cd_day, cd_night)

    numerator = constants.RECIPROCAL_LAMBDA_20C * delta * (rn - g) + gamma * (cn / (t + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + cd * u2)
    
    pet_pm = numerator / denominator
    pet_pm = np.clip(pet_pm, 0.0, None)  

    return xr.DataArray(
        pet_pm,
        dims=temperature.dims,
        coords=temperature.coords,
        name="pet_penman_monteith_hourly_mm",
    )
