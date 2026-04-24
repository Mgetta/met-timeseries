"""
Source-agnostic derived variable calculations.

Each function is a pure transformation on :class:`xarray.DataArray` inputs —
no knowledge of NLDAS/PRISM variable names or :class:`xarray.Dataset` structure.
The caller maps source-specific variable names to function arguments.

Derivations happen on the grid **before** spatial aggregation because nonlinear
functions do not commute with spatial averaging:
``mean(sqrt(u² + v²)) ≠ sqrt(mean(u)² + mean(v)²)``.
"""

from __future__ import annotations

import datetime

import pandas as pd
import numpy as np
import xarray as xr
import pvlib
import pyet


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Compute wind speed from U and V components using MetPy.

    Parameters
    ----------
    u:
        U-component of wind in m/s.
    v:
        V-component of wind in m/s.

    Returns
    -------
    :class:`xarray.DataArray` named ``wind_speed_ms``.
    """
    from metpy.calc import wind_speed as _mpc_wind_speed
    from metpy.units import units

    u_q = u.values * units("m/s")
    v_q = v.values * units("m/s")
    ws = _mpc_wind_speed(u_q, v_q)
    return xr.DataArray(
        ws.magnitude,
        dims=u.dims,
        coords=u.coords,
        name="wind_speed_ms",
    )


def adjust_wind_height(
    wind_speed: xr.DataArray,
    z_from: float = 10.0,
    z_to: float = 2.0,
    method: str = "logarithmic",
    alpha: float = 1 / 7,
) -> xr.DataArray:
    """Adjust wind speed between measurement heights.

    Two methods are available:

    **Logarithmic** (FAO-56 Eq. 47):

    .. math::

        u_{z_{to}} = u_{z_{from}} \\times
        \\frac{\\ln(67.8\\, z_{to} - 5.42)}{\\ln(67.8\\, z_{from} - 5.42)}

    Derived from Monin-Obukhov similarity theory under neutral stability,
    with an implicit roughness length of ~0.01 m (short grass).
    Reference: Allen et al. (1998), FAO Irrigation & Drainage Paper 56.

    **Power law** (Hellman / Davenport):

    .. math::

        u_{z_{to}} = u_{z_{from}} \\times
        \\left(\\frac{z_{to}}{z_{from}}\\right)^{\\alpha}

    where α is a surface-roughness exponent.  Common values:

    =============================  =====
    Surface type                   α
    =============================  =====
    Open water / ice               0.10
    Open flat terrain (1/7 rule)   0.143
    Agricultural crops             0.20
    Suburban / forest              0.25
    Urban                          0.35
    =============================  =====

    Reference: Davenport (1960); Justus & Mikhail (1976).

    Parameters
    ----------
    wind_speed:
        Wind speed in m/s at height ``z_from``.
    z_from:
        Source measurement height in metres. Default is 10 m (NLDAS).
    z_to:
        Target height in metres. Default is 2 m (FAO-56 standard).
    method:
        ``"logarithmic"`` (FAO-56) or ``"power_law"`` (Hellman).
    alpha:
        Power-law exponent. Only used when ``method="power_law"``.
        Default is 1/7 ≈ 0.143 (open flat terrain).

    Returns
    -------
    :class:`xarray.DataArray` named ``wind_speed_ms`` adjusted to ``z_to``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"logarithmic"`` or ``"power_law"``.
    """
    if method == "logarithmic":
        # FAO-56 Eq. 47: u_2 = u_z × 4.87 / ln(67.8z - 5.42)
        # Generalised for arbitrary z_to:
        #   u_z_to = u_z_from × ln(67.8·z_to - 5.42) / ln(67.8·z_from - 5.42)
        factor = np.log(67.8 * z_to - 5.42) / np.log(67.8 * z_from - 5.42)
        adjusted = wind_speed * factor

    elif method == "power_law":
        # Hellman / Davenport: u_z_to = u_z_from × (z_to / z_from)^α
        adjusted = wind_speed * (z_to / z_from) ** alpha

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'logarithmic' or 'power_law'."
        )

    return adjusted.rename("wind_speed_ms")


# ---------------------------------------------------------------------------
# Humidity
# ---------------------------------------------------------------------------

def dewpoint_from_specific_humidity(
    specific_humidity: xr.DataArray,
    pressure: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute dewpoint temperature from specific humidity.

    Uses ``metpy.calc.dewpoint_from_specific_humidity`` for the full
    thermodynamic chain: specific humidity → mixing ratio → vapour pressure →
    dewpoint, which is more accurate than a simple Magnus approximation.

    Parameters
    ----------
    specific_humidity:
        Specific humidity in kg/kg.
    pressure:
        Atmospheric pressure in Pa.  Defaults to standard atmosphere
        (101 325 Pa) when not provided.

    Returns
    -------
    :class:`xarray.DataArray` named ``dewpoint_c`` in degrees Celsius.
    """
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
    """Compute dewpoint from specific humidity using Clausius-Clapeyron.

    Replicates the methodology from the original MetTool:

    1. Specific humidity → mixing ratio: ``w = q / (1 − q)``
    2. Mixing ratio → actual vapor pressure: ``eₐ = w / (0.622 + w) × P``
    3. Clausius-Clapeyron inversion:
       ``Td = 1 / (1/273.15 − 0.0001844 × ln(eₐ / 0.6113))``

    The constant ``0.0001844`` derives from ``Rᵥ / Lᵥ`` where
    ``Rᵥ = 461.5 J/(kg·K)`` (specific gas constant for water vapor) and
    ``Lᵥ ≈ 2.501e6 J/kg`` (latent heat of vaporization at 0°C):
    ``Rᵥ / Lᵥ = 461.5 / 2501000 ≈ 0.0001844``.

    The reference vapor pressure ``0.6113 kPa`` is the saturation vapor
    pressure at 0°C (273.15 K) from the Clausius-Clapeyron relation.

    Parameters
    ----------
    specific_humidity:
        Specific humidity in kg/kg.
    pressure:
        Atmospheric pressure in Pa.

    Returns
    -------
    :class:`xarray.DataArray` named ``dewpoint_c`` in degrees Celsius.
    """
    q = specific_humidity.values.astype(float)
    p_kpa = pressure.values.astype(float) / 1000.0  # Pa → kPa

    # Step 1: specific humidity → mixing ratio
    w = q / (1.0 - q)

    # Step 2: mixing ratio → actual vapor pressure (kPa)
    e_a = (w / (0.622 + w)) * p_kpa

    # Step 3: Clausius-Clapeyron inversion → dewpoint (K)
    REF_VP = 0.6113  # saturation VP at 0°C (kPa)
    CC_CONST = 0.0001844  # Rv / Lv (K⁻¹)

    with np.errstate(divide="ignore", invalid="ignore"):
        td_k = 1.0 / (1.0 / 273.15 - CC_CONST * np.log(e_a / REF_VP))

    # K → °C
    td_c = td_k - 273.15

    return xr.DataArray(
        td_c,
        dims=specific_humidity.dims,
        coords=specific_humidity.coords,
        name="dewpoint_c",
    )

def dewpoint_august_roche_magnus(
    temperature: xr.DataArray,  # °C
    specific_humidity: xr.DataArray,  # kg/kg
    pressure: xr.DataArray,  # Pa
) -> xr.DataArray:
    """Compute dewpoint temperature using the August-Roche-Magnus equation.

    Parameters
    ----------
    temperature:
        Air temperature in degrees Celsius.
    specific_humidity:
        Specific humidity in kg/kg.
    pressure:
        Atmospheric pressure in Pa.

    Returns
    -------
    :class:`xarray.DataArray` named ``dewpoint_c``.
        Dewpoint temperature in degrees Celsius.

    Notes
    -----
    This uses the standard constants for water vapor:
    - A = 17.27
    - B = 237.7 (°C)
    The formula assumes ambient pressure (e.g., sea level). For applications
    at altitude, corrections for vapor pressure may be needed.

    References
    ----------
    Magnus-Tetens approximation for vapor pressure:
    https://en.wikipedia.org/wiki/Dew_point#Calculating_the_dew_point
    """
    A = 17.27
    B = 237.7  # °C

    relative_humidity = calculate_relative_humidity(temperature, specific_humidity, pressure)  # Assume sea level pressure

    # Saturation vapor pressure (e_s) in hPa
    e_s = 6.112 * np.exp((A * temperature) / (B + temperature))

    # Actual vapor pressure (e), proportional to RH
    e = (relative_humidity / 100) * e_s

    # Dewpoint temperature (T_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        dewpoint = (B * np.log(e / 6.112)) / (A - np.log(e / 6.112))

    return xr.DataArray(
        dewpoint,
        dims=temperature.dims,
        coords=temperature.coords,
        name="dewpoint_c",
    )

def kelvin_to_fahrenheit(temp_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Fahrenheit.

    Parameters
    ----------
    temp_k:
        Temperature in Kelvin.

    Returns
    -------
    :class:`xarray.DataArray` named ``temp_f`` in degrees Fahrenheit.
    """
    return ((temp_k - 273.15) * 9/5 + 32).rename("temp_f")


def kelvin_to_celsius(temp_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    temp_k:
        Temperature in Kelvin.

    Returns
    -------
    :class:`xarray.DataArray` named ``temp_c`` in degrees Celsius.
    """
    return (temp_k - 273.15).rename("temp_c")


def cloud_cover_davis(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
    k: float = 0.65,
    min_clearsky: float = 10.0,
) -> xr.DataArray:
    """Estimate cloud-cover fraction using the Davis (1975) method.

    Derives cloud fraction from the clearness index (observed / clear-sky
    shortwave radiation) via the Angström-Prescott relationship:

    .. math::

        \\frac{R_s}{R_{so}} = 1 - k \\, C^2

    Inverted for cloud fraction *C*:

    .. math::

        C = \\sqrt{\\frac{1 - R_s / R_{so}}{k}}

    Compared to the simple linear method (``cloud_cover``), this produces
    lower cloud fractions under partly-cloudy conditions and saturates at
    ``C = 1`` when ``R_s/R_{so} \\leq 1 - k``.

    Parameters
    ----------
    shortwave:
        Observed downward shortwave radiation in W/m².  Expected dims are
        ``(time, lat, lon)`` when *time* is provided, or ``(lat, lon)``.
    lat:
        Latitude values in decimal degrees.
    lon:
        Longitude values in decimal degrees.
    time:
        Array of datetime-like values for solar geometry.  When ``None``
        a fixed reference time (2000-07-01 12:00 UTC) is used.
    k:
        Angström-Prescott cloud-opacity coefficient.  Default ``0.65``
        is typical for mid-latitude continental sites (Davis et al., 1975).
        Lower values (e.g. 0.50) suit humid/tropical climates; higher
        values (e.g. 0.75) suit arid regions.

    Returns
    -------
    :class:`xarray.DataArray` named ``cloud_cover_fraction_davis``.
        Values are in [0, 1].  Nighttime pixels (clear-sky < 10 W/m²)
        are NaN.

    References
    ----------
    Davis, J.A., et al. (1975). "Estimation of solar radiation."
    Agricultural Meteorology, 15(3), 355–366.
    """
    # Build the clear-sky radiation array (reuse existing helper)
    clearsky, daytime_mask = calculate_clearsky_array(shortwave, lat, lon, time,min_solar_elevation=min_clearsky)

    with np.errstate(divide="ignore", invalid="ignore"):
        clearness_index = shortwave / clearsky

    # Clamp clearness index to [0, 1] — observed can exceed clear-sky
    # due to cloud-edge enhancement or model/obs mismatch
    clearness_index = clearness_index.clip(0.0, 1.0)

    # Davis inversion: C = sqrt((1 - Kt) / k)
    with np.errstate(invalid="ignore"):
        cc = np.sqrt((1.0 - clearness_index) / k)

    cc = cc.clip(0.0, 1.0)
    cc = cc.where(daytime_mask)
    return cc.rename("cloud_cover_fraction_davis")


def thompson_sky_cover_radiation(
    clear_sky_rad: xr.DataArray,
    cloud_cover: xr.DataArray,
    b_coef: float = 0.3
) -> xr.DataArray:
    """
    Computes actual solar radiation from sky cover based on Thompson (1976).
    
    Equation: Rs = Rso * [B + (1 - B) * (1 - N)^0.61]
    
    Parameters
    ----------
    clear_sky_rad : xr.DataArray
        Clear-sky solar radiation (Rso).
    cloud_cover : xr.DataArray
        Fractional cloud cover (N), valid range is 0.0 to 1.0.
        If your data is in tenths (0-10) or percent (0-100), convert to 0-1 first.
    b_coef : float
        Empirical coefficient for transmission under fully overcast skies.
        Default is a generic 0.3, but should ideally be calibrated per station.
        
    Returns
    -------
    xr.DataArray
        Estimated global solar radiation (Rs).
    """
    # Ensure cloud cover is strictly bounded between 0 and 1 to prevent invalid math
    N = cloud_cover.clip(min=0.0, max=1.0)
    
    # Thompson's empirical cloud attenuation factor
    cloud_factor = b_coef + (1.0 - b_coef) * (1.0 - N)**0.61
    
    # Calculate actual surface solar radiation
    rs = clear_sky_rad * cloud_factor
    
    # Preserve metadata cleanly
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
    """Estimate cloud-cover fraction using Thompson's (1976) parabolic method.

    Fits cloud cover as a second-order polynomial of the clearness index
    ``Kt = Rs / Rso``:

    .. math::

        C = a + b \\, K_t + c \\, K_t^2

    The default coefficients are constrained so that:

    - ``C(Kt=1.0) = 0`` — clear sky yields zero cloud cover
    - ``C(Kt=0.2) = 1`` — dense overcast (diffuse floor ~20% of Rso)
    - ``C(Kt=0.6) ≈ 0.5`` — partly cloudy midpoint

    These satisfy ``a + b + c = 0`` and ``a + 0.2b + 0.04c = 1``.

    Parameters
    ----------
    shortwave:
        Observed downward shortwave radiation in W/m².
    lat:
        Latitude values in decimal degrees.
    lon:
        Longitude values in decimal degrees.
    time:
        Datetime-like values for solar geometry.
    min_solar_elevation:
        Minimum solar elevation angle in degrees.  Default 10°.
    coeffs:
        Parabolic coefficients ``(a, b, c)``.  Default values are derived
        from the boundary conditions above.  Site-specific calibration
        against observed cloud cover data is recommended.

    Returns
    -------
    :class:`xarray.DataArray` named ``cloud_cover_fraction_thompson``.
        Values are clamped to [0, 1].  Nighttime pixels are NaN.

    References
    ----------
    Thompson, O.E. (1976). "Climatological models for estimating solar
    radiation." *Proceedings, Workshop on Solar Energy and the Biosphere*.
    """
    from numpy.polynomial import polynomial as P

    clearsky, daytime_mask = calculate_clearsky_array(
        shortwave, lat, lon, time, min_solar_elevation,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        kt = (shortwave / clearsky).clip(0.0, 1.0)


    if coeffs is None:
        coeffs = np.polyfit(kt_obs, cc_obs, deg=2)  # returns [c, b, a]
        coeffs = (coeffs[2], coeffs[1], coeffs[0])

    a, b, c = coeffs



    # Parabolic equation: C = a + b·Kt + c·Kt²
    cc = a + b * kt + c * kt ** 2



    cc = cc.clip(0.0, 1.0)
    cc = cc.where(daytime_mask)
    return cc.rename("cloud_cover_fraction_thompson")

def cloud_cover_linear(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
    min_clearsky: float = 10.0,
) -> xr.DataArray:
    """Estimate cloud-cover fraction (linear method).

    ``cloud_cover = 1 - observed / clear_sky``, clamped to [0, 1].
    Nighttime pixels are NaN.
    """
    clearsky, daytime_mask = calculate_clearsky_array(shortwave, lat, lon, time,min_solar_elevation=min_clearsky)


    with np.errstate(divide="ignore", invalid="ignore"):
        cc = 1.0 - shortwave / clearsky

    cc = cc.clip(0.0, 1.0)
    cc = cc.where(daytime_mask)
    return cc.rename("cloud_cover_fraction")

def hamon_weiss_wilson_clearsky(
    day_of_year: xr.DataArray,
    latitude: xr.DataArray
) -> xr.DataArray:
    """
    Computes daily clear-sky solar radiation (Rso) approximating the 
    Hamon, Weiss, and Wilson (1954) latitude and seasonal curves.
    
    Parameters
    ----------
    day_of_year : xr.DataArray
        Day of the year (1-366).
    latitude : xr.DataArray
        Latitude in decimal degrees.
        
    Returns
    -------
    xr.DataArray
        Clear-sky solar radiation (Rso) in Langleys/day 
        (the standard historical unit used in the 1954 paper).
    """
    # Convert latitude to radians
    phi = np.radians(latitude)
    
    # Fractional year in radians
    theta = 2.0 * np.pi * day_of_year / 365.0
    
    # Solar declination angle (delta) in radians
    delta = 0.006918 - 0.399912 * np.cos(theta) + 0.070257 * np.sin(theta) \
            - 0.006758 * np.cos(2 * theta) + 0.000907 * np.sin(2 * theta) \
            - 0.002697 * np.cos(3 * theta) + 0.00148 * np.sin(3 * theta)
            
    # Sunset hour angle (omega_s) in radians
    omega_s = np.arccos((-np.tan(phi) * np.tan(delta)).clip(min=-1.0, max=1.0))
    
    # Eccentricity correction factor of Earth's orbit
    dr = 1.000110 + 0.034221 * np.cos(theta) + 0.001280 * np.sin(theta) \
         + 0.000719 * np.cos(2 * theta) + 0.000077 * np.sin(2 * theta)
         
    # Solar Constant in Langleys/day (approx. 1.94 Langleys/min * 1440 mins)
    # Using Langleys as it is the native unit for the 1954 method.
    Gsc = 2793.6 
    
    # Extraterrestrial radiation (Ra) in Langleys/day
    Ra = (Gsc / np.pi) * dr * (
        omega_s * np.sin(phi) * np.sin(delta) +
        np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    
    # The Hamon-Weiss-Wilson curves generally translate to a clear-sky 
    # atmospheric transmissivity factor. For US Latitudes, 0.73 is the 
    # classic mean bulk coefficient used to represent their clear-sky nomograms.
    Rso = 0.73 * Ra
    
    Rso.attrs['long_name'] = 'Clear-sky Solar Radiation (Hamon-Weiss-Wilson parameterization)'
    Rso.attrs['units'] = 'Langleys/day'
    
    return Rso


def hamon_weiss_wilson_actual_radiation(
    clear_sky_rad: xr.DataArray,
    percent_sunshine: xr.DataArray,
    a_coef: float = 0.22,
    b_coef: float = 0.78
) -> xr.DataArray:
    """
    Computes actual incident solar radiation using the Hamon, Weiss, 
    and Wilson (1954) empirical relationship based on percent sunshine.
    
    Parameters
    ----------
    clear_sky_rad : xr.DataArray
        Clear-sky solar radiation (Rso).
    percent_sunshine : xr.DataArray
        Percent of possible sunshine duration (valid range: 0.0 to 100.0).
    a_coef : float
        Fraction of radiation occurring on a completely overcast day (0% sunshine).
    b_coef : float
        Coefficient representing radiation scaling with sunshine.
        (Note: a_coef + b_coef typically equals ~1.0).
        
    Returns
    -------
    xr.DataArray
        Estimated actual global solar radiation (Rs).
    """
    # Bound percent sunshine between 0 and 100, then convert to a fraction (0 to 1)
    sun_frac = percent_sunshine.clip(min=0.0, max=100.0) / 100.0
    
    # Apply the linear sunshine/insolation empirical formula
    rs = clear_sky_rad * (a_coef + b_coef * sun_frac)
    
    rs.attrs['long_name'] = 'Estimated Global Solar Radiation'
    rs.attrs['method'] = 'Hamon, Weiss, Wilson (1954) Sunshine Duration'
    if 'units' in clear_sky_rad.attrs:
        rs.attrs['units'] = clear_sky_rad.attrs['units']
        
    return rs

# ---------------------------------------------------------------------------
# PET
# ---------------------------------------------------------------------------


def pet_hargreaves(
    daily_tmin,
    daily_tmax,
    lat: float,
):
    """Estimate daily PET using the Hargreaves (1985) method.

    Uses ``mettoolbox.utils.radiation`` for extraterrestrial radiation (Ra)
    computed from latitude and day-of-year.

    Formula:
        ``PET = 0.0023 × Ra × (T_mean + 17.8) × sqrt(T_max - T_min)``

    where Ra is in MJ/m²/day and PET is in mm/day.

    Parameters
    ----------
    daily_tmin:
        Daily minimum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    daily_tmax:
        Daily maximum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    lat:
        Latitude in decimal degrees.

    Returns
    -------
    :class:`pandas.Series` named ``pet_hargreaves_mm``.
    """
    import pandas as pd
    from mettoolbox.utils import radiation

    tmean = (daily_tmin + daily_tmax) / 2.0
    trange = (daily_tmax - daily_tmin).clip(lower=0.0)

    # mettoolbox.utils.radiation expects a Series/DataFrame with a DatetimeIndex
    ra_df = radiation(tmean.to_frame(name="temp"), lat)
    ra = ra_df["ra"]

    with np.errstate(invalid="ignore"):
        pet = 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange)

    pet = pet.fillna(0.0).clip(lower=0.0)
    pet.name = "pet_hargreaves_mm"
    return pet

def pevt_penman_pyet_daily(
    shortwave_hourly: xr.DataArray,   # W/m²
    temperature_hourly: xr.DataArray, # °C
    dewpoint_hourly: xr.DataArray,    # °C
    wind_hourly: xr.DataArray,        # m/s at 2 m
    elevation: "xr.DataArray | float" = 0.0,  # m
    albedo: float = 0.06,
) -> xr.DataArray:
    """Daily Penman pan evaporation on a spatial grid via pyet (mm/day).

    Aggregates hourly inputs to daily then calls ``pyet.penman`` directly on
    the xarray inputs.  Use ``albedo=0.06`` for open-water / pan (Kohler et al. 1955).
    """
    rs_daily    = (shortwave_hourly * 0.0036).resample(time="1D").sum()  # W/m² → MJ/m²/day
    tmean_daily = temperature_hourly.resample(time="1D").mean()          # °C
    wind_daily  = wind_hourly.resample(time="1D").mean()                 # m/s
    ea_daily    = (
        0.6108 * np.exp(17.27 * dewpoint_hourly / (dewpoint_hourly + 237.3))
    ).resample(time="1D").mean()                                          # kPa

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
    shortwave_hourly: xr.DataArray,   # W/m²
    temperature_hourly: xr.DataArray, # °C
    dewpoint_hourly: xr.DataArray,    # °C — pass two options for two PEVT variants
    wind_hourly: xr.DataArray,        # m/s at 2 m
    elevation: "xr.DataArray | float" = 0.0,  # m
    albedo: float = 0.06,
    min_solar_elevation: float = 10.0,
) -> xr.DataArray:
    """Hourly Penman pan evaporation on a spatial grid (mm/hr).

    Computes daily PET via ``pevt_penman_pyet_daily`` then disaggregates to
    hourly proportional to daytime observed shortwave.  Nighttime = 0.

    Call twice with different ``dewpoint_hourly`` arrays to produce the two
    PEVT variants (Liston & Elder 2006).
    """
    pet_daily = pevt_penman_pyet_daily(
        shortwave_hourly, temperature_hourly, dewpoint_hourly,
        wind_hourly, elevation, albedo,
    )  # mm/day

    _, daytime_mask = calculate_clearsky_array(
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
    shortwave_hourly: xr.DataArray,   # W/m²
    temperature_hourly: xr.DataArray, # °C
    dewpoint_hourly: xr.DataArray,    # °C — two options → two PEVT variants
    wind_hourly: xr.DataArray,        # m/s at 2 m
    pressure: xr.DataArray | float = 1013.25,        # hPa
    albedo: float = 0.06,
    min_solar_elevation: float = 10.0,
) -> xr.DataArray:
    """Hourly Penman pan evaporation via Kohler et al. (1955) (mm/hr).

    Aggregates hourly inputs to daily (SOLR MJ/m²/day, mean ATEM °C, mean
    DEWP °C, mean wind m/s), applies the Penman/Kohler formula, then
    disaggregates back to hourly proportional to daytime observed shortwave.
    Nighttime = 0.

    Call twice with different ``dewpoint_hourly`` to produce the two PEVT
    variants (Liston & Elder 2006).  Use ``albedo=0.06`` for open-water / pan.
    """
    # --- Aggregate to daily ---
    rs_daily    = (shortwave_hourly * 0.0036).resample(time="1D").sum()  # W/m² → MJ/m²/day
    tmean_daily = temperature_hourly.resample(time="1D").mean()          # °C
    dp_daily    = dewpoint_hourly.resample(time="1D").mean()             # °C
    wind_daily  = wind_hourly.resample(time="1D").mean()                 # m/s

    # --- Penman / Kohler formula on daily inputs ---
    es       = 6.112 * np.exp(17.67 * tmean_daily / (tmean_daily + 243.5))  # hPa
    ea       = 6.112 * np.exp(17.67 * dp_daily    / (dp_daily    + 243.5))  # hPa
    vpd      = (es - ea).clip(min=0.0)
    delta    = 4098.0 * es / (tmean_daily + 237.3) ** 2                      # hPa/°C
    gamma    = 0.000665 * pressure                                            # hPa/°C
    lambda_v = 2.501 - 0.002361 * tmean_daily                                # MJ/kg
    rn       = rs_daily * (1.0 - albedo)                                      # MJ/m²/day
    f_u      = 0.005 + 0.00085 * (wind_daily * 3.6)                          # Kohler wind fn (km/hr)
    ea_term  = f_u * vpd

    pet_daily = ((delta * (rn / lambda_v) + gamma * ea_term) / (delta + gamma)).clip(min=0.0)

    # --- Disaggregate daily → hourly via daytime SW pattern ---
    _, daytime_mask = calculate_clearsky_array(
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
    shortwave_down: xr.DataArray,  # W/m²
    longwave_down: xr.DataArray,  # W/m²
    temperature: xr.DataArray,  # K
    dewpoint: xr.DataArray,  # K
    wind_speed: xr.DataArray,  # m/s
    pressure: xr.DataArray,  # Pa
    elevation: float = 0.0,  # Elevation in meters
    albedo: float = 0.23,  # Default albedo
) -> xr.DataArray:
    """Compute hourly Penman Pan Evaporation (mm/hour).

    Parameters
    ----------
    shortwave_down:
        Downward shortwave radiation in W/m².
    longwave_down:
        Downward longwave radiation in W/m².
    temperature:
        Air temperature in Kelvin (K).
    dewpoint:
        Dewpoint temperature in Kelvin (K).
    wind_speed:
        Wind speed at 2 m in m/s.
    pressure:
        Surface pressure in Pa.
    elevation:
        Elevation above sea level in meters.
    albedo:
        Surface albedo (default 0.23 for grassland).

    Returns
    -------
    Hourly Penman Pan Evaporation as xarray.DataArray in mm/hour.
    """
    # Constants
    cn = 37.0  # Wind function coefficient
    cd = 0.24  # Resistance coefficient for hourly FAO-56 adjustment

    # Step 1: Compute net radiation (R_n) in MJ/m²/hour (convert W/m² to MJ/m²/hr)
    net_radiation = calculate_net_radiation(shortwave_down, longwave_down, temperature, albedo)
    net_radiation_mj = net_radiation * 0.0036  # W/m² → MJ/m²/hour

    # Step 2: Compute specific humidity (q) from dewpoint and pressure
    specific_humidity = calculate_specific_humidity(temperature, dewpoint, pressure)

    # Step 3: Compute aerodynamic term (vapor pressure deficit)
    # Saturation and actual vapor pressure (kPa) based on temperature
    def _vapor_pressure(temp_k):
        return 0.6108 * np.exp(17.27 * (temp_k - 273.15) / (temp_k - 273.15 + 237.3))
    e_s = _vapor_pressure(temperature)
    e_a = _vapor_pressure(dewpoint)

    # Step 4: Psychrometric constant (γ)
    # Atmospheric pressure (Pa → kPa)
    pressure_kpa = pressure / 1000.0
    gamma = 0.000665 * pressure_kpa  # Psychrometric constant (kPa/°C)

    # Step 5: Compute Penman Pan Evaporation
    delta = (4098 * e_s) / ((temperature - 273.15) + 237.3) ** 2  # Slope of saturation VP curve
    vapor_pressure_deficit = e_s - e_a

    numerator = 0.408 * delta * net_radiation_mj + gamma * cn / (temperature - 273.15 + 273) * wind_speed * vapor_pressure_deficit
    denominator = delta + gamma * (1 + cd * wind_speed)

    pet_hourly = numerator / denominator  # mm/hour
    pet_hourly = xr.DataArray(
        pet_hourly.clip(0.0),  # Ensure non-negative evaporation
        dims=shortwave_down.dims,
        coords=shortwave_down.coords,
        name="pet_penman_hourly_mm",
    )
    return pet_hourly

def pet_penman_monteith_hourly(
    temperature: xr.DataArray,  # °C
    wind_speed: xr.DataArray,  # m/s
    shortwave: xr.DataArray,  # W/m²
    dewpoint: xr.DataArray,  # °C
    elevation: "xr.DataArray | float",  # m
) -> xr.DataArray:
    """Compute hourly FAO-56 Penman-Monteith reference ET (mm/hour).

    Parameters
    ----------
    temperature:
        Instantaneous air temperature in °C, dims (time, lat, lon).
    wind_speed:
        Wind speed at 2 m height in m/s, dims (time, lat, lon).
    shortwave:
        Incoming shortwave radiation in W/m² (converted internally to MJ/m²/hour),
        dims (time, lat, lon).
    dewpoint:
        Dewpoint temperature in °C, dims (time, lat, lon).
    elevation:
        Elevation above sea level in metres. May be a scalar float or a gridded
        xarray.DataArray broadcastable to the spatial dimensions.

    Returns
    -------
    :class:`xarray.DataArray`
        Hourly ET in mm/hour with dims matching the inputs.
    """
    # Constants
    cn = 37.0    # Wind function numerator (hourly FAO-56 coefficient)
    cd_day = 0.24
    cd_night = 0.96
    sigma_h = 2.042e-10  # Stefan-Boltzmann MJ/(m²·h·K⁴)
    humid_a = 0.34
    humid_b = 0.14
    albedo = 0.23  # Grassland reference value
    
    # Convert inputs to raw NumPy arrays
    t = temperature.values.astype(float)
    u2 = wind_speed.values.astype(float)
    rs_wm2 = shortwave.values.astype(float)
    td = dewpoint.values.astype(float)
    z = elevation.values.astype(float) if isinstance(elevation, xr.DataArray) else float(elevation)

    # Convert shortwave radiation to MJ/m²/hour (W/m² → MJ/m²/hour)
    rs = rs_wm2 * 0.0036

    # Atmospheric pressure (kPa) [Eq. 7]
    pressure = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26  # (kPa)

    # Psychrometric constant (γ, kPa/°C) [Eq. 8]
    gamma = 0.000665 * pressure

    # Saturation vapor pressure (eₛ, kPa) at hourly temperature [Eq. 11]
    def _magnus(temp):
        return 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    es = _magnus(t)

    # Actual vapor pressure (eₐ, kPa) from dewpoint [Eq. 14]
    ea = _magnus(td)

    # Slope of saturation vapor pressure curve (Δ, kPa/°C) [Eq. 13]
    delta = 4098.0 * es / (t + 237.3) ** 2

    # Net shortwave radiation (Rns, MJ/m²/hour) [Eq. 38]
    rns = (1.0 - albedo) * rs

    # Net longwave radiation (Rnl, MJ/m²/hour) [Eq. 39]
    t_k = t + 273.15
    rnl = sigma_h * (t_k**4) * (humid_a - humid_b * np.sqrt(np.maximum(ea, 0))) * rs

    # Net radiation (Rn, MJ/m²/hour) [Eq. 40]
    rn = rns - rnl

    # Soil heat flux G: (0.1 * Rn daytime, 0.5 * Rn nighttime)
    is_daytime = rn > 0.0
    g = np.where(is_daytime, 0.1 * rn, 0.5 * rn)

    # Wind resistance coefficient Cd
    cd = np.where(is_daytime, cd_day, cd_night)

    # Penman-Monteith hourly ET [Eq. 53]
    numerator = 0.408 * delta * (rn - g) + gamma * (cn / (t + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + cd * u2)
    pet_pm = numerator / denominator
    pet_pm = np.clip(pet_pm, 0.0, None)  # Avoid negative ET

    return xr.DataArray(
        pet_pm,
        dims=temperature.dims,
        coords=temperature.coords,
        name="pet_penman_monteith_hourly_mm",
    )
# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


STEFAN_BOLTZMANN = 5.67e-8  # W/m²/K⁴

def calculate_net_radiation(
    shortwave_down: xr.DataArray,  # W/m² - Downward solar radiation
    longwave_down: xr.DataArray,  # W/m² - Downward longwave radiation
    temperature: xr.DataArray,  # K - Surface temperature
    albedo: float = 0.23,  # Default albedo for grassland
    emissivity: float = 0.97,  # Default surface emissivity
) -> xr.DataArray:
    """Calculate net radiation (R_n) in W/m².

    Parameters
    ----------
    shortwave_down:
        Downward shortwave solar radiation (W/m²).
    longwave_down:
        Downward longwave atmospheric radiation (W/m²).
    temperature:
        Surface temperature in Kelvin (K).
    albedo:
        Surface albedo for reflectance (default 0.23 for grassland).
    emissivity:
        Surface emissivity (default 0.97 for most land surfaces).

    Returns
    -------
    Net radiation (R_n) as xarray.DataArray in W/m².
    """
    # Stefan-Boltzmann constant (W/m²/K⁴)
    sigma = STEFAN_BOLTZMANN

    # Reflected shortwave radiation
    shortwave_up = shortwave_down * albedo

    # Outgoing longwave radiation (Stefan-Boltzmann law)
    longwave_up = emissivity * sigma * temperature**4

    # Net radiation: combine all components
    net_radiation = (shortwave_down - shortwave_up) + (longwave_down - longwave_up)

    return net_radiation.rename("net_radiation")



# Saturation vapor pressure (e_s) in kPa
def svp_magnus(temperature: xr.DataArray, A=17.27, B=237.3):
    '''
    Calculate saturation vapor pressure (eₛ) in kPa using the Magnus formula.
    Parameters
    ----------
    temperature:
        Air temperature in celsius.
    A, B:
        Constants for water vapor over water.

    Returns
    -------
    :class:`xarray.DataArray`
        Saturation vapor pressure in kPa.
    '''
    # A, B Constants for water vapor over water
    return 0.6108 * np.exp(A * (temperature) / (temperature + B))

def calculate_specific_humidity(
    temperature: xr.DataArray,  # K
    dewpoint: xr.DataArray,  # K
    pressure: xr.DataArray,  # Pa
    epsilon: float = 0.622,  # Ratio of molecular weights of water vapor/dry air
) -> xr.DataArray:
    """Calculate specific humidity (q) in kg/kg.

    Parameters
    ----------
    temperature:
        Air temperature in Celsius (C).
    dewpoint:
        Dewpoint temperature in Celsius (C).
    pressure:
        Surface pressure in kPa.

    Returns
    -------
    Specific humidity (kg/kg) as xarray.DataArray.
    """
    # Convert pressure from Pa to kPa
    pressure_kpa = pressure


    e_s = svp_magnus(temperature)  # Saturation vapor pressure at air temperature

    # Actual vapor pressure (e_a) from dewpoint
    e_a = svp_magnus(dewpoint)

    # Specific humidity (q) calculation
    q = (epsilon * e_a) / (pressure_kpa - (1 - epsilon) * e_a)

    return q.rename("specific_humidity")

def calculate_clearsky_radiation(lat: float, lon: float, dt: datetime.datetime) -> float:
    """Compute theoretical clear-sky surface shortwave radiation (W/m²).

    Uses standard solar geometry with a fixed atmospheric transmittance.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees.
    lon:
        Longitude in decimal degrees (retained for future equation-of-time
        corrections).
    dt:
        UTC datetime for which to compute the radiation.

    Returns
    -------
    Clear-sky surface shortwave radiation in W/m², or 0.0 at nighttime.
    """
    S0 = 1361.0  # solar constant (W/m²)
    tau = 0.75   # clear-sky atmospheric transmittance

    n = dt.timetuple().tm_yday  # day of year

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

    return float(S0 * cos_zenith * tau)


def calculate_clearsky_array(
    shortwave: xr.DataArray,
    lat: xr.DataArray | np.ndarray,
    lon: xr.DataArray | np.ndarray,
    time: xr.DataArray | np.ndarray | None = None,
    min_solar_elevation: float = 10.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build clear-sky radiation and daytime mask arrays using pvlib.

    Uses the Ineichen/Perez clear-sky model with the built-in Linke
    turbidity climatology.  This accounts for seasonal and geographic
    variation in atmospheric clarity (aerosols, water vapor) without
    requiring any inputs beyond lat/lon/time.

    The Linke turbidity climatology is a 0.5° monthly global dataset
    derived from satellite observations.  At NLDAS resolution (~12 km)
    the spatial sampling is adequate.  pvlib interpolates between months
    automatically.

    Parameters
    ----------
    shortwave:
        Observed shortwave, used only for shape/coords.
    lat, lon:
        Coordinate arrays in decimal degrees.
    time:
        Datetime-like values for solar geometry.
    min_solar_elevation:
        Minimum solar elevation angle in degrees.  Default 10°.

    Returns
    -------
    (clearsky, daytime_mask)
    """

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

            # get_solarposition + get_clearsky handles airmass internally
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
 

 
def calculate_relative_humidity(
    temperature: xr.DataArray,  # °C
    specific_humidity: xr.DataArray,  # kg/kg
    pressure: xr.DataArray,  # Pa
) -> xr.DataArray:
    """Compute relative humidity from specific humidity, pressure, and temperature.

    Parameters
    ----------
    temperature:
        Air temperature in °C.
    specific_humidity:
        Specific humidity (dimensionless ratio, kg/kg).
    pressure:
        Surface pressure in Pa.

    Returns
    -------
    :class:`xarray.DataArray` named ``relative_humidity``.
        Relative humidity in %.
    """
    # Constants
    epsilon = 0.622  # Ratio of molecular weight of water vapor to dry air
    A = 17.27
    B = 237.7  # °C

    # Convert pressure from Pa to hPa
    pressure_hpa = pressure / 100.0

    # Saturation vapor pressure (e_s) in hPa
    e_s = 6.112 * np.exp((A * temperature) / (B + temperature))

    # Actual vapor pressure (e) in hPa
    e = (specific_humidity * pressure_hpa) / (epsilon + specific_humidity)

    # Relative Humidity (RH)
    rh = (e / e_s) * 100.0

    return xr.DataArray(
        rh.clip(0.0, 100.0),  # Clamp to [0%, 100%]
        dims=temperature.dims,
        coords=temperature.coords,
        name="relative_humidity",
    )