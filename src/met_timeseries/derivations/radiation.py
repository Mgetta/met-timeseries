from __future__ import annotations

import datetime
import pandas as pd
import numpy as np
import xarray as xr
import pvlib
from met_timeseries.derivations import constants, humidity


SOLAR_CONSTANT = 1361.0 # W/m² at Earth's distance, used in geometric clearsky model and as base for pvlib calculations

# --- Clearsky Transmittance ---
CLEARSKY_TRANSMITTANCE = 0.75  # Geometric clearsky model (clearsky_radiation_geometric)
CLEARSKY_TRANSMITTANCE_HWW    = 0.73  # HWW (1954) clearsky model (clearsky_radiation_hww)
                                       # NB: different empirical source to SIMPLE

# --- Brutsaert Net Longwave Emissivity (Bruin 1987) ---
BRUTSAERT_A = 0.34  # Atmospheric emissivity coefficient a
BRUTSAERT_B = 0.14  # Atmospheric emissivity coefficient b


def net_radiation(
    shortwave_down: xr.DataArray,
    longwave_down: xr.DataArray,
    temperature: xr.DataArray,
    albedo: float = 0.23,
    emissivity: float = 0.97,
) -> xr.DataArray:
    """Calculate net radiation (R_n) in W/m².

    Args:
        temperature: Surface temperature in °C (converted to K internally for
            the Stefan-Boltzmann longwave-up term).
    """
    shortwave_up = shortwave_down * albedo
    temperature_k = temperature + 273.15  # °C → K for Stefan-Boltzmann
    longwave_up = emissivity * constants.STEFAN_BOLTZMANN * temperature_k**4
    net_radiation = (shortwave_down - shortwave_up) + (longwave_down - longwave_up)
    return net_radiation.rename("net_radiation")

def clearsky_radiation_geometric(lat: float, lon: float, dt: datetime.datetime) -> float:
    """Compute theoretical clear-sky surface shortwave radiation (W/m²)."""
      

    n = dt.timetuple().tm_yday  
    dec_deg = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + n)))
    dec_rad = np.radians(dec_deg)

    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    omega_rad = np.radians(15.0 * (hour - 12.0))
    lat_rad = np.radians(lat)

    cos_zenith = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(omega_rad)
    )

    if cos_zenith <= 0.0:
        return 0.0

    return float(SOLAR_CONSTANT_W_M2 * cos_zenith * CLEARSKY_TRANSMITTANCE)

def clearsky_radiation_ineichen(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
    min_solar_elevation: float = 10.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build clear-sky radiation and daytime mask arrays using pvlib."""
    lat_vals = np.asarray(lat)
    lon_vals = np.asarray(lon)

    if time is not None:
        times = pd.DatetimeIndex(np.asarray(time))
    else:
        times = pd.DatetimeIndex([datetime.datetime(2000, 7, 1, 12, 0, 0)])

    clearsky_values = np.full(shortwave.shape, np.nan, dtype=float)
    elevation_values = np.full(shortwave.shape, np.nan, dtype=float)

    time_axis = shortwave.dims.index("time") if "time" in shortwave.dims else None

    for i, la in enumerate(lat_vals):
        for j, lo in enumerate(lon_vals):
            loc = pvlib.location.Location(latitude=la, longitude=lo)
            solpos = loc.get_solarposition(times)
            cs = loc.get_clearsky(times, model="ineichen", solar_position=solpos)

            if time_axis is not None:
                clearsky_values[:, i, j] = cs["ghi"].values
                elevation_values[:, i, j] = solpos["apparent_elevation"].values
            else:
                clearsky_values[i, j] = cs["ghi"].iloc[0]
                elevation_values[i, j] = solpos["apparent_elevation"].iloc[0]    
     
    clearsky = xr.DataArray(clearsky_values, dims=shortwave.dims, coords=shortwave.coords)
    elevation = xr.DataArray(elevation_values, dims=shortwave.dims, coords=shortwave.coords)

    daytime_mask = elevation >= min_solar_elevation
    return clearsky, daytime_mask


def clearsky_radiation_hww(
    day_of_year: xr.DataArray,
    latitude: xr.DataArray
) -> xr.DataArray:
    """Computes daily clear-sky solar radiation approximating Hamon, Weiss, Wilson (1954)."""
    phi = np.radians(latitude)
    theta = 2.0 * np.pi * day_of_year / 365.0
    delta = 0.006918 - 0.399912 * np.cos(theta) + 0.070257 * np.sin(theta) \
            - 0.006758 * np.cos(2 * theta) + 0.000907 * np.sin(2 * theta) \
            - 0.002697 * np.cos(3 * theta) + 0.00148 * np.sin(3 * theta)
            
    omega_s = np.arccos((-np.tan(phi) * np.tan(delta)).clip(min=-1.0, max=1.0))
    dr = 1.000110 + 0.034221 * np.cos(theta) + 0.001280 * np.sin(theta) \
         + 0.000719 * np.cos(2 * theta) + 0.000077 * np.sin(2 * theta)
         
    Gsc = 117.5  # MJ/m²/day (= 2793.6 Langleys/day × 0.04184 MJ/Langley)
    Ra = (Gsc / np.pi) * dr * (
        omega_s * np.sin(phi) * np.sin(delta) +
        np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    
    Rso = 0.73 * Ra
    Rso.attrs['long_name'] = 'Clear-sky Solar Radiation (Hamon-Weiss-Wilson parameterization)'
    Rso.attrs['units'] = 'MJ/m²/day'
    return Rso







def cloud_cover_davis(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
    k: float = 0.65,
    min_clearsky: float = 10.0,
) -> xr.DataArray:
    """Estimate cloud-cover fraction using the Davis (1975) method."""
    clearsky, daytime_mask = clearsky_radiation_ineichen(
        shortwave, lat, lon, time, min_solar_elevation=min_clearsky
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        clearness_index = (shortwave / clearsky).clip(0.0, 1.0)

    with np.errstate(invalid="ignore"):
        cc = np.sqrt((1.0 - clearness_index) / k)

    cc = cc.clip(0.0, 1.0).where(daytime_mask)
    return cc.rename("cloud_cover_fraction_davis")

def sky_cover_radiation_thompson(
    clear_sky_rad: xr.DataArray,
    cloud_cover: xr.DataArray,
    b_coef: float = 0.3
) -> xr.DataArray:
    """Computes actual solar radiation from sky cover based on Thompson (1976)."""
    N = cloud_cover.clip(min=0.0, max=1.0)
    cloud_factor = b_coef + (1.0 - b_coef) * (1.0 - N)**0.61
    rs = clear_sky_rad * cloud_factor
    rs.attrs['long_name'] = 'Estimated Global Solar Radiation'
    rs.attrs['method'] = 'Thompson (1976) Sky Cover Attenuation'
    if 'units' in clear_sky_rad.attrs:
        rs.attrs['units'] = clear_sky_rad.attrs['units']
    return rs

def cloud_cover_thompson(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
    min_solar_elevation: float = 10.0,
    coeffs: tuple[float, float, float] | None = None,
) -> xr.DataArray:
    """Estimate cloud-cover fraction using Thompson's (1976) parabolic method."""
    clearsky, daytime_mask = clearsky_array(
        shortwave, lat, lon, time, min_solar_elevation,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = (shortwave / clearsky).clip(0.0, 1.0)

    # Defaults to constraint bounds if custom empirical arrays are not provided
    if coeffs is None:
        coeffs = (0.0, 1.0, 0.0)  # Safe fallback (linear equivalent), update logically per context

    a, b, c = coeffs
    cc = a + b * kt + c * kt ** 2
    cc = cc.clip(0.0, 1.0).where(daytime_mask)
    return cc.rename("cloud_cover_fraction_thompson")

def cloud_cover_linear(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
    min_clearsky: float = 10.0,
) -> xr.DataArray:
    """Estimate cloud-cover fraction (linear method)."""
    clearsky, daytime_mask = clearsky_radiation_ineichen(
        shortwave, lat, lon, time, min_solar_elevation=min_clearsky
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        cc = 1.0 - shortwave / clearsky

    cc = cc.clip(0.0, 1.0).where(daytime_mask)
    return cc.rename("cloud_cover_fraction")


def actual_radiation_hww(
    clear_sky_rad: xr.DataArray,
    percent_sunshine: xr.DataArray,
    a_coef: float = 0.22,
    b_coef: float = 0.78
) -> xr.DataArray:
    """Computes actual incident solar radiation using Hamon, Weiss, and Wilson (1954)."""
    sun_frac = percent_sunshine.clip(min=0.0, max=100.0) / 100.0
    rs = clear_sky_rad * (a_coef + b_coef * sun_frac)
    rs.attrs['long_name'] = 'Estimated Global Solar Radiation'
    rs.attrs['method'] = 'Hamon, Weiss, Wilson (1954) Sunshine Duration'
    if 'units' in clear_sky_rad.attrs:
        rs.attrs['units'] = clear_sky_rad.attrs['units']
    return rs

def net_longwave_brutsaert(
    temperature: xr.DataArray,        # °C
    vapor_pressure: xr.DataArray,     # kPa  (actual, not saturation)
    clearsky_shortwave: xr.DataArray | None = None,
    shortwave_down: xr.DataArray | None = None,
    humid_a: float = 0.34,
    humid_b: float = 0.14,
) -> xr.DataArray:
    """
    Estimate net longwave radiation loss (W/m²) using Brutsaert's atmospheric
    emissivity parameterisation.

    R_nl = σ T⁴ · (a - b√eₐ) · (Rs/Rso)

    The cloud correction term (Rs/Rso) requires both shortwave_down and
    clearsky_shortwave. If either is None the term is omitted (clear-sky assumed).

    Args:
        temperature:        Air temperature (°C)
        vapor_pressure:     Actual vapor pressure (kPa) — pass actual, not saturation.
                            Use humidity.saturation_vapor_pressure_arm(dewpoint) to derive.
        clearsky_shortwave: Clear-sky shortwave (W/m²) for cloud correction.
        shortwave_down:     Observed shortwave (W/m²) for cloud correction.
        humid_a:            Emissivity coefficient a (default 0.34, Bruin 1987)
        humid_b:            Emissivity coefficient b (default 0.14, Bruin 1987)

    Returns:
        Net longwave radiation in W/m² (positive = upward loss).
    """
    t_k = temperature + 273.15

    if clearsky_shortwave is not None and shortwave_down is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rs_rso = (shortwave_down / clearsky_shortwave).clip(0.25, 1.0)
    else:
        rs_rso = xr.ones_like(temperature)

    rnl = (
        constants.STEFAN_BOLTZMANN
        * t_k**4
        * (humid_a - humid_b * np.sqrt(vapor_pressure.clip(min=0.0)))
        * rs_rso
    )
    return rnl.rename("net_longwave_brutsaert")


def net_radiation_arm(
    shortwave_down: xr.DataArray,
    temperature: xr.DataArray,        # °C
    dewpoint: xr.DataArray,           # °C
    clearsky_shortwave: xr.DataArray | None = None,
    albedo: float = 0.23,
    humid_a: float = 0.34,
    humid_b: float = 0.14,
) -> xr.DataArray:
    """
    Estimate net radiation (W/m²) using the August-Roche-Magnus vapor pressure
    approximation for atmospheric emissivity (Brutsaert-style net longwave).

    Suitable when longwave_down is unavailable and must be parameterised from
    temperature and dewpoint. Delegates to humidity.saturation_vapor_pressure_arm
    and net_longwave_brutsaert.

    Args:
        shortwave_down:     Incoming shortwave radiation (W/m²)
        temperature:        Air temperature (°C)
        dewpoint:           Dewpoint temperature (°C)
        clearsky_shortwave: Clear-sky shortwave (W/m²) for cloud correction.
                            If None, clear-sky is assumed (Rs/Rso = 1).
        albedo:             Surface albedo (default 0.23, short reference grass)
        humid_a:            Emissivity coefficient a (default 0.34)
        humid_b:            Emissivity coefficient b (default 0.14)
    """
    e_a = humidity.vapor_pressure_magnus(
        dewpoint,
        b=constants.VAPOR_B_TETENS,
        base=0.6112,
    )
    rns = shortwave_down * (1.0 - albedo)

    rnl = net_longwave_brutsaert(
        temperature,
        vapor_pressure=e_a,
        clearsky_shortwave=clearsky_shortwave,
        shortwave_down=shortwave_down,
        humid_a=humid_a,
        humid_b=humid_b,
    )

    return (rns - rnl).rename("net_radiation_arm")