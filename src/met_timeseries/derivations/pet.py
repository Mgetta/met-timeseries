from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pyet

from met_timeseries.derivations import thermodynamics
from met_timeseries.derivations import radiation


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

def pevt_penman_pyet_daily(
    shortwave_hourly: xr.DataArray,
    temperature_hourly: xr.DataArray,
    dewpoint_hourly: xr.DataArray,
    wind_hourly: xr.DataArray,
    elevation: "xr.DataArray | float" = 0.0,
    albedo: float = 0.06,
) -> xr.DataArray:
    """Daily Penman pan evaporation on a spatial grid via pyet (mm/day)."""
    rs_daily    = (shortwave_hourly * 0.0036).resample(time="1D").sum()  
    tmean_daily = temperature_hourly.resample(time="1D").mean()          
    wind_daily  = wind_hourly.resample(time="1D").mean()                 
    ea_daily    = (
        0.6108 * np.exp(17.27 * dewpoint_hourly / (dewpoint_hourly + 237.3))
    ).resample(time="1D").mean()                                          

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

    return pet.clip(min=0.0).rename("pevt_penman_mm_day").assign_attrs(
        {"units": "mm/day", "albedo": albedo}
    )

def pevt_penman_pyet_hourly(
    shortwave_hourly: xr.DataArray,
    temperature_hourly: xr.DataArray,
    dewpoint_hourly: xr.DataArray,
    wind_hourly: xr.DataArray,
    elevation: "xr.DataArray | float" = 0.0,
    albedo: float = 0.06,
    min_solar_elevation: float = 10.0,
) -> xr.DataArray:
    """Hourly Penman pan evaporation on a spatial grid (mm/hr)."""
    pet_daily = pevt_penman_pyet_daily(
        shortwave_hourly, temperature_hourly, dewpoint_hourly,
        wind_hourly, elevation, albedo,
    )

    _, daytime_mask = radiation.calculate_clearsky_array(
        shortwave_hourly,
        lat=shortwave_hourly.coords["lat"].values,
        lon=shortwave_hourly.coords["lon"].values,
        time=shortwave_hourly.coords["time"].values,
        min_solar_elevation=min_solar_elevation,
    )

    sw_daytime   = shortwave_hourly.where(daytime_mask, 0.0)
    sw_daily_sum = sw_daytime.resample(time="1D").sum()
    sw_daily_sum = sw_daily_sum.where(sw_daily_sum > 0.0, other=np.nan)

    pet_broadcast = pet_daily.reindex_like(shortwave_hourly, method="ffill")
    sw_sum_bcast  = sw_daily_sum.reindex_like(shortwave_hourly, method="ffill")

    pet_hourly = (pet_broadcast * sw_daytime / sw_sum_bcast).fillna(0.0).clip(min=0.0)

    return pet_hourly.rename("pevt_penman_mm_hr").assign_attrs(
        {"units": "mm/hr", "albedo": albedo}
    )

def pevt_penman_kohler(
    shortwave_hourly: xr.DataArray,
    temperature_hourly: xr.DataArray,
    dewpoint_hourly: xr.DataArray,
    wind_hourly: xr.DataArray,
    pressure: xr.DataArray | float = 1013.25,
    albedo: float = 0.06,
    min_solar_elevation: float = 10.0,
) -> xr.DataArray:
    """Hourly Penman pan evaporation via Kohler et al. (1955) (mm/hr)."""
    rs_daily    = (shortwave_hourly * 0.0036).resample(time="1D").sum() 
    tmean_daily = temperature_hourly.resample(time="1D").mean()          
    dp_daily    = dewpoint_hourly.resample(time="1D").mean()             
    wind_daily  = wind_hourly.resample(time="1D").mean()                 

    es       = 6.112 * np.exp(17.67 * tmean_daily / (tmean_daily + 243.5))  
    ea       = 6.112 * np.exp(17.67 * dp_daily    / (dp_daily    + 243.5))  
    vpd      = (es - ea).clip(min=0.0)
    delta    = 4098.0 * es / (tmean_daily + 237.3) ** 2                      
    gamma    = 0.000665 * pressure                                            
    lambda_v = 2.501 - 0.002361 * tmean_daily                                
    rn       = rs_daily * (1.0 - albedo)                                      
    f_u      = 0.005 + 0.00085 * (wind_daily * 3.6)                          
    ea_term  = f_u * vpd

    pet_daily = ((delta * (rn / lambda_v) + gamma * ea_term) / (delta + gamma)).clip(min=0.0)

    _, daytime_mask = radiation.calculate_clearsky_array(
        shortwave_hourly,
        lat=shortwave_hourly.coords["lat"].values,
        lon=shortwave_hourly.coords["lon"].values,
        time=shortwave_hourly.coords["time"].values,
        min_solar_elevation=min_solar_elevation,
    )

    sw_daytime   = shortwave_hourly.where(daytime_mask, 0.0)
    sw_daily_sum = sw_daytime.resample(time="1D").sum()
    sw_daily_sum = sw_daily_sum.where(sw_daily_sum > 0.0, other=np.nan)

    pet_broadcast = pet_daily.reindex_like(shortwave_hourly, method="ffill")
    sw_sum_bcast  = sw_daily_sum.reindex_like(shortwave_hourly, method="ffill")

    pet_hourly = (pet_broadcast * sw_daytime / sw_sum_bcast).fillna(0.0).clip(min=0.0)

    return pet_hourly.rename("pevt_penman_kohler_mm_hr").assign_attrs(
        {"units": "mm/hr", "albedo": albedo}
    )

def pet_penman_hourly(
    shortwave_down: xr.DataArray,
    longwave_down: xr.DataArray,
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
    wind_speed: xr.DataArray,
    pressure: xr.DataArray,
    elevation: float = 0.0,
    albedo: float = 0.23,
) -> xr.DataArray:
    """Compute hourly Penman Pan Evaporation (mm/hour)."""
    cn = 37.0  
    cd = 0.24  

    net_radiation = thermodynamics.calculate_net_radiation(shortwave_down, longwave_down, temperature, albedo)
    net_radiation_mj = net_radiation * 0.0036  

    specific_humidity = thermodynamics.calculate_specific_humidity(temperature, dewpoint, pressure)

    def _vapor_pressure(temp_k):
        return 0.6108 * np.exp(17.27 * (temp_k - 273.15) / (temp_k - 273.15 + 237.3))
        
    e_s = _vapor_pressure(temperature)
    e_a = _vapor_pressure(dewpoint)

    pressure_kpa = pressure / 1000.0
    gamma = 0.000665 * pressure_kpa  

    delta = (4098 * e_s) / ((temperature - 273.15) + 237.3) ** 2  
    vapor_pressure_deficit = e_s - e_a

    numerator = 0.408 * delta * net_radiation_mj + gamma * cn / (temperature - 273.15 + 273) * wind_speed * vapor_pressure_deficit
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
    elevation: "xr.DataArray | float",
) -> xr.DataArray:
    """Compute hourly FAO-56 Penman-Monteith reference ET (mm/hour)."""
    cn = 37.0    
    cd_day = 0.24
    cd_night = 0.96
    sigma_h = 2.042e-10  
    humid_a = 0.34
    humid_b = 0.14
    albedo = 0.23  
    
    t = temperature.values.astype(float)
    u2 = wind_speed.values.astype(float)
    rs_wm2 = shortwave.values.astype(float)
    td = dewpoint.values.astype(float)
    z = elevation.values.astype(float) if isinstance(elevation, xr.DataArray) else float(elevation)

    rs = rs_wm2 * 0.0036
    pressure = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26  
    gamma = 0.000665 * pressure

    def _magnus(temp):
        return 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    es = _magnus(t)
    ea = _magnus(td)

    delta = 4098.0 * es / (t + 237.3) ** 2
    rns = (1.0 - albedo) * rs
    
    t_k = t + 273.15
    rnl = sigma_h * (t_k**4) * (humid_a - humid_b * np.sqrt(np.maximum(ea, 0))) * rs

    rn = rns - rnl
    is_daytime = rn > 0.0
    g = np.where(is_daytime, 0.1 * rn, 0.5 * rn)
    cd = np.where(is_daytime, cd_day, cd_night)

    numerator = 0.408 * delta * (rn - g) + gamma * (cn / (t + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + cd * u2)
    
    pet_pm = numerator / denominator
    pet_pm = np.clip(pet_pm, 0.0, None)  

    return xr.DataArray(
        pet_pm,
        dims=temperature.dims,
        coords=temperature.coords,
        name="pet_penman_monteith_hourly_mm",
    )