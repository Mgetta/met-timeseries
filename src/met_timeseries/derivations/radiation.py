from __future__ import annotations

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

def clearsky_radiation_geometric(datarray: xr.DataArray) -> xr.DataArray:
    """Compute theoretical clear-sky surface shortwave radiation (W/m²).

    Args:
        shortwave: datarray with dims including 'time', 'lat', 'lon'.
                   Used only for its coordinates/shape as a template.

    Returns:
        datarray of the same shape as shortwave with clear-sky radiation values.
    """
    lat_vals = datarray.coords["lat"].values
    times = pd.DatetimeIndex(datarray.coords["time"].values)

    clearsky_values = np.zeros(datarray.shape, dtype=float)

    for t_idx, dt in enumerate(times):
        n = dt.timetuple().tm_yday
        dec_deg = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + n)))
        dec_rad = np.radians(dec_deg)

        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        omega_rad = np.radians(15.0 * (hour - 12.0))
        lat_rad = np.radians(lat_vals)

        cos_zenith = (
            np.sin(lat_rad) * np.sin(dec_rad)
            + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(omega_rad)
        )
        cos_zenith = np.maximum(cos_zenith, 0.0)  # shape (Nlat,)

        cs_at_time = SOLAR_CONSTANT * cos_zenith * CLEARSKY_TRANSMITTANCE  # (Nlat,)
        clearsky_values[t_idx, :, :] = cs_at_time[:, np.newaxis]

    return xr.DataArray(clearsky_values, dims=datarray.dims, coords=datarray.coords)

def clearsky_radiation_ineichen(
    dataarray: xr.DataArray,
) -> xr.DataArray:
    """Build a clear-sky radiation array using the pvlib lat/lon Ineichen model.

    Args:
        shortwave: DataArray with dims including 'time', 'lat', 'lon'.
                   Used only for its coordinates/shape as a template.

    Returns:
        Clear-sky GHI DataArray with the same dims/coords as shortwave.
    """
    lat_vals = dataarray.coords["lat"].values
    lon_vals = dataarray.coords["lon"].values
    times = pd.DatetimeIndex(dataarray.coords["time"].values)

    clearsky_values = np.full(dataarray.shape, np.nan, dtype=float)

    n_lat = len(lat_vals)
    n_lon = len(lon_vals)

    for idx in range(n_lat * n_lon):
        i, j = np.unravel_index(idx, (n_lat, n_lon))
        la, lo = lat_vals[i], lon_vals[j]
        loc = pvlib.location.Location(latitude=la, longitude=lo)
        solpos = loc.get_solarposition(times)
        cs = loc.get_clearsky(times, model="ineichen", solar_position=solpos)
        clearsky_values[:, i, j] = cs["ghi"].values

    return xr.DataArray(clearsky_values, dims=dataarray.dims, coords=dataarray.coords)

def lazy_clearsky_ineichen(shortwave: xr.DataArray) -> xr.DataArray:
    """Wraps the eager pvlib clearsky function for lazy Dask evaluation."""
    
    return xr.apply_ufunc(
        clearsky_radiation_ineichen, # Your existing pvlib wrapper function
        shortwave,                   # The lazy Dask input
        dask="parallelized",         # The magic keyword that prevents RAM spikes
        output_dtypes=[float],       # Crucial: Dask needs to know what type comes out without computing it
        # If your function needs to operate along a specific dimension (like time), 
        # uncomment and adjust the line below:
        # vectorize=True 
    )

def extra_radiation(
    hourly_template: xr.DataArray
) -> xr.DataArray:
    """
    Computes daily extraterrestrial radiation (Ra) at the top of the atmosphere.
    
    Args:
        hourly_template: DataArray used purely to extract 'time' and 'lat' coordinates.
        
    Returns:
        xr.DataArray: Daily Ra in MJ/m²/day.
    """
    # 1. Extract coordinates
    lat = hourly_template.coords["lat"]
    lat_rad = np.radians(lat)
    
    # We use daily time, so resample to daily if the template is hourly
    daily_time = hourly_template.coords["time"].resample(time="1D").first()
    day_of_year = daily_time.dt.dayofyear
    
    # 2. Solar Geometry Variables
    # Solar Constant in MJ/m²/min is roughly 0.0820, which is 118.08 MJ/m²/day
    # (Note: FAO-56 uses Gsc = 0.0820 MJ/m²/min)
    Gsc = 0.0820 * 60 * 24  # MJ/m²/day (approx 118.08)
    
    # Inverse relative distance Earth-Sun (dr)
    theta = (2.0 * np.pi / 365.0) * day_of_year
    dr = 1.0 + 0.033 * np.cos(theta)
    
    # Solar declination (delta)
    delta = 0.409 * np.sin(theta - 1.39)
    
    # Sunset hour angle (omega_s)
    # Clip prevents domain errors during polar night/day
    x = (-np.tan(lat_rad) * np.tan(delta)).clip(min=-1.0, max=1.0)
    omega_s = np.arccos(x)
    
    # 3. Calculate Ra (FAO-56 Equation 21)
    # The integration of solar geometry over the course of the day
    ra_mj = (Gsc / np.pi) * dr * (
        omega_s * np.sin(lat_rad) * np.sin(delta) +
        np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )
    
    # 4. Broadcast to the correct daily dimensions
    # Using np.newaxis or xarray broadcasting to ensure it aligns with (time, lat)
    ra_mj = ra_mj.transpose("time", "lat")
    
    return ra_mj.rename("extraterrestrial_radiation_mj_day").assign_attrs(
        {"units": "MJ/m²/day", "long_name": "Extraterrestrial Solar Radiation"}
    )


def extra_radiation_pvlib(
    hourly_template: xr.DataArray
) -> xr.DataArray:
    """
    Computes daily Extraterrestrial Radiation (Ra) in MJ/m²/day using pvlib.
    
    This method calculates hourly horizontal extra-terrestrial radiation 
    and then integrates it over the day.
    """
    lat_vals = hourly_template.coords["lat"].values
    lon_vals = hourly_template.coords["lon"].values
    times = pd.DatetimeIndex(hourly_template.coords["time"].values)
    
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)
    
    # 1. Get Top-of-Atmosphere Direct Normal Irradiance (DNI_extra)
    # This only depends on the day of the year (Earth-Sun distance), 
    # so we only need to calculate it once for the whole time series.
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values  # Shape: (time,)
    
    # Array to hold our hourly horizontal radiation in W/m²
    ra_hourly_values = np.zeros((len(times), n_lat, n_lon), dtype=float)
    
    # 2. Loop through spatial grid (the pvlib constraint)
    for idx in range(n_lat * n_lon):
        i, j = np.unravel_index(idx, (n_lat, n_lon))
        la, lo = lat_vals[i], lon_vals[j]
        
        # Get solar position for this specific pixel
        loc = pvlib.location.Location(latitude=la, longitude=lo)
        solpos = loc.get_solarposition(times)
        
        # Zenith angle (0 is straight up, 90 is horizon)
        zenith_rad = np.radians(solpos['zenith'].values)
        
        # 3. Project DNI onto the Horizontal Plane
        # Ra = DNI_extra * cos(zenith). Clip at 0 so nights don't go negative.
        cos_zenith = np.maximum(np.cos(zenith_rad), 0.0)
        
        # Broadcast DNI_extra across the zenith array
        ra_hourly_values[:, i, j] = dni_extra * cos_zenith

    # 4. Rebuild the xarray DataArray for the hourly values (W/m²)
    ra_hourly_w = xr.DataArray(
        ra_hourly_values, 
        dims=hourly_template.dims, 
        coords=hourly_template.coords
    )
    
    # 5. Convert to Daily Volume
    # Convert W/m² to MJ/m²/hr (multiply by 0.0036), then sum the 24 hours
    ra_daily_mj = (ra_hourly_w * 0.0036).resample(time="1D").sum()
    
    return ra_daily_mj.rename("extraterrestrial_radiation_pvlib_mj_day")

def clearsky_radiation_hww(datarray: xr.DataArray) -> xr.DataArray:
    """Computes daily clear-sky solar radiation approximating Hamon, Weiss, Wilson (1954).

    Args:
        datarray: DataArray with dims including 'time' and 'lat'.
                   Used only for its coordinates/shape as a template.

    Returns:
        DataArray of clear-sky radiation (MJ/m²/day) broadcast to the same
        dims/coords as shortwave.
    """
    day_of_year = datarray.coords["time"].dt.dayofyear
    latitude = datarray.coords["lat"]
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
    Rso_broadcast = Rso.broadcast_like(datarray)
    Rso_broadcast.attrs['long_name'] = 'Clear-sky Solar Radiation (Hamon-Weiss-Wilson parameterization)'
    Rso_broadcast.attrs['units'] = 'W/m²/day'
    return Rso_broadcast

def clearsky_radiation_fao56(
    ra: xr.DataArray,     # Extraterrestrial radiation (Ra)
    elevation: float      # Station elevation in meters (z)
) -> xr.DataArray:
    """
    Calculates clear-sky solar radiation (Rso) using FAO-56 Eq 37.
    """
    
    # Calculate the elevation-based atmospheric transmissivity
    transmissivity = 0.75 + (2e-5 * elevation)
    
    # Calculate Rso
    rso = transmissivity * ra
    
    return rso.rename("clearsky_shortwave_fao56")

def daytime_mask_solar_elevation(
    dataarray: xr.DataArray,
    min_solar_elevation: float = 10.0,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.DataArray:
    """Return a boolean DataArray that is True during daytime hours.

    Daytime is defined as times when the apparent solar elevation exceeds
    `min_solar_elevation` degrees, computed via pvlib for each grid point.

    Args:
        dataarray: DataArray with dims including 'time', 'lat', 'lon'.
                   Used only for its coordinates/shape as a template.
        min_solar_elevation: Minimum apparent solar elevation angle (degrees)
                             above which a timestep is considered daytime.

    Returns:
        Boolean DataArray with the same dims/coords as shortwave.
    """
    lat_vals = dataarray.coords[lat_coord].values
    lon_vals = dataarray.coords[lon_coord].values
    times = pd.DatetimeIndex(dataarray.coords["time"].values)

    elevation_values = np.full(dataarray.shape, np.nan, dtype=float)

    n_lat = len(lat_vals)
    n_lon = len(lon_vals)

    for idx in range(n_lat * n_lon):
        i, j = np.unravel_index(idx, (n_lat, n_lon))
        la, lo = lat_vals[i], lon_vals[j]
        loc = pvlib.location.Location(latitude=la, longitude=lo)
        solpos = loc.get_solarposition(times)
        elevation_values[:, i, j] = solpos["apparent_elevation"].values

    elevation = xr.DataArray(elevation_values, dims=dataarray.dims, coords=dataarray.coords)
    return elevation >= min_solar_elevation





def cloud_cover_davis(
    shortwave: xr.DataArray,
    k: float = 0.65,
    min_clearsky: float = 10.0,
) -> xr.DataArray:
    """Estimate cloud-cover fraction using the Davis (1975) method."""
    clearsky = _clearsky_radiation_ineichen(shortwave)
    daytime_mask = _daytime_mask_solar_elevation(shortwave, min_solar_elevation=min_clearsky)

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
    min_solar_elevation: float = 10.0,
    coeffs: tuple[float, float, float] | None = None,
) -> xr.DataArray:
    """Estimate cloud-cover fraction using Thompson's (1976) parabolic method."""
    clearsky = _clearsky_radiation_ineichen(shortwave)
    daytime_mask = _daytime_mask_solar_elevation(shortwave, min_solar_elevation=min_solar_elevation)

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
    min_clearsky: float = 10.0,
) -> xr.DataArray:
    """Estimate cloud-cover fraction (linear method)."""
    clearsky = _clearsky_radiation_ineichen(shortwave)
    daytime_mask = _daytime_mask_solar_elevation(shortwave, min_solar_elevation=min_clearsky)

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


def cloud_factor_ratio(
    shortwave_hourly: xr.DataArray,
    clearsky_shortwave_hourly: xr.DataArray
) -> xr.DataArray:
    
    # 1. Sum radiation over the whole day
    rs_daily = shortwave_hourly.resample(time="1D").sum()
    rso_daily = clearsky_shortwave_hourly.resample(time="1D").sum()
    
    # 2. Calculate the ratio on the daily totals (avoids the nighttime 0/0 issue)
    rs_rso_daily = (rs_daily / rso_daily).clip(0.25, 1.0)
    
    # 3. Calculate the daily cloud factor
    cloud_factor_daily = 1.35 * rs_rso_daily - 0.35
    
    # 4. Broadcast/forward-fill that daily factor back to the hourly timestep
    cloud_factor_hourly = cloud_factor_daily.resample(time="1h").ffill()
    
    return cloud_factor_hourly


def net_longwave_direct(
    temperature: xr.DataArray,        # °C (from NLDAS)
    downward_longwave: xr.DataArray,  # W/m² (NLDAS DLWRF)
    surface_emissivity: float = 0.98  # Standard for open water / wet soil
) -> xr.DataArray:
    """
    Calculates Net Outgoing Longwave radiation directly using measured/modeled 
    downward longwave radiation, bypassing empirical cloud corrections.
    """
    t_k = temperature + 273.15
    
    # 1. Calculate Upward Longwave emitted by the surface
    r_lu = surface_emissivity * constants.STEFAN_BOLTZMANN * t_k**4
    
    # 2. Calculate Net Outgoing Longwave
    # (Positive means energy is leaving the surface, which is standard for PET)
    r_nl = r_lu - downward_longwave
    
    return r_nl.rename("net_outgoing_longwave")

def net_longwave_brutsaert(
        temperature: xr.DataArray,        # °C
        vapor_pressure: xr.DataArray,     # kPa  (actual, not saturation)
        surface_emissivity: float = 0.98
) -> xr.DataArray:
    
    t_k = temperature + 273.15
    vp = vapor_pressure.clip(min=0.0) * 10 # Prevent negative vapor pressure values causing NaNs in emissivity
    
    atmos_emissivity = 1.24 * (vp / t_k) ** (1.0 / 7.0) 
    dlr = atmos_emissivity * constants.STEFAN_BOLTZMANN * t_k**4

    r_lu = surface_emissivity * constants.STEFAN_BOLTZMANN * t_k**4

    rnl = r_lu - dlr
    return rnl.rename("net_longwave_brutsaert")

def net_longwave_brunt(
    temperature: xr.DataArray,        # °C
    vapor_pressure: xr.DataArray,     # kPa  (actual, not saturation)
    clearsky_shortwave: xr.DataArray | None = None,
    shortwave_down: xr.DataArray | None = None,
    humid_a: float = 0.34,
    humid_b: float = 0.14,
) -> xr.DataArray:
    
    t_k = temperature + 273.15
    vp = vapor_pressure.clip(min=0.0)# Prevent negative vapor pressure values causing NaNs in emissivity
    
    if clearsky_shortwave is not None and shortwave_down is not None:
        cloud_factor = cloud_factor_ratio(shortwave_down, clearsky_shortwave)
    else:
        cloud_factor = xr.ones_like(t_k)

    rnl = (
        constants.STEFAN_BOLTZMANN
        * t_k**4
        * (humid_a - humid_b * np.sqrt(vp))
        * cloud_factor
    )
    
    return rnl

def cloud_factor_fao56(
    shortwave: xr.DataArray,
    clearsky_shortwave) -> xr.DataArray:
    # Prevent 0/0 division at night by using xr.where. 
    # When clearsky is 0 (night), assume a clear-sky ratio of 1.0
    rs_rso = xr.where(
        clearsky_shortwave > 0, 
        shortwave / clearsky_shortwave, 
        1.0
    )
    # Clip the ratio to physically valid bounds per FAO-56
    rs_rso = rs_rso.clip(0.25, 1.0)
    
    # Apply the actual FAO-56 Eq 39 cloudiness correction factor
    cloud_factor = 1.35 * rs_rso - 0.35
    return cloud_factor


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

    rnl = net_longwave_brunt(
        temperature,
        vapor_pressure=e_a,
        clearsky_shortwave=clearsky_shortwave,
        shortwave_down=shortwave_down,
        humid_a=humid_a,
        humid_b=humid_b,
    )

    return (rns - rnl).rename("net_radiation_arm")



def _ineichen_1d(lat: float, lon: float, times: pd.DatetimeIndex) -> np.ndarray:
    """Core PVLib calculation for clear-sky radiation at a single point."""
    loc = pvlib.location.Location(latitude=lat, longitude=lon)
    cs = loc.get_clearsky(times, model="ineichen")
    return cs["ghi"].values

def _solar_elevation_1d(lat: float, lon: float, times: pd.DatetimeIndex) -> np.ndarray:
    """Core PVLib calculation for solar elevation at a single point."""
    loc = pvlib.location.Location(latitude=lat, longitude=lon)
    solpos = loc.get_solarposition(times)
    return solpos["apparent_elevation"].values


# --- 2. The Dask-Friendly Xarray Wrappers ---

def _clearsky_radiation_ineichen(dataarray: xr.DataArray) -> xr.DataArray:
    """
    Build a clear-sky radiation array lazily.
    Automatically adapts to 2D grids (lat, lon) or 1D arrays (polygon_id).
    """
    times_pd = pd.DatetimeIndex(dataarray.coords["time"].values)
    
    # Broadcast handles the geometry. 
    # If inputs are 1D polygon centroids, it yields 1D arrays. 
    # If inputs are lat/lon vectors, it builds a 2D mesh grid.
    lat_grid, lon_grid = xr.broadcast(dataarray.coords["lat"], dataarray.coords["lon"])

    clearsky_da = xr.apply_ufunc(
        _ineichen_1d,
        lat_grid,
        lon_grid,
        kwargs={"times": times_pd},
        input_core_dims=[[], []],    # Lat and Lon are scalars per point
        output_core_dims=[['time']], # The output adds a time dimension
        vectorize=True,              # Let Dask handle the looping
        dask="parallelized",
        output_dtypes=[float]
    )

    # Reorder dimensions to exactly match the input template
    return clearsky_da.transpose(*dataarray.dims)

def _daytime_mask_solar_elevation(
    dataarray: xr.DataArray,
    min_solar_elevation: float = 10.0,
    lat_coord: str = "lat",
    lon_coord: str = "lon",
) -> xr.DataArray:
    """
    Lazily compute a boolean daytime mask based on solar elevation.
    """
    times_pd = pd.DatetimeIndex(dataarray.coords["time"].values)
    lat_grid, lon_grid = xr.broadcast(dataarray.coords[lat_coord], dataarray.coords[lon_coord])

    elevation_da = xr.apply_ufunc(
        _solar_elevation_1d,
        lat_grid,
        lon_grid,
        kwargs={"times": times_pd},
        input_core_dims=[[], []],
        output_core_dims=[['time']],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )

    elevation_da = elevation_da.transpose(*dataarray.dims)
    return elevation_da >= min_solar_elevation