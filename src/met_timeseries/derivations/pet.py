"""FAO-56 Penman-Monteith hourly reference evapotranspiration."""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_pet_fao56(
    temp_k: xr.DataArray | np.ndarray,
    dewpoint_k: xr.DataArray | np.ndarray,
    wind_speed_2m: xr.DataArray | np.ndarray,
    solar_rad: xr.DataArray | np.ndarray,
    pressure: xr.DataArray | np.ndarray,
    lat: xr.DataArray | np.ndarray | None = None,
    doy: xr.DataArray | np.ndarray | None = None,
) -> xr.DataArray | np.ndarray:
    """Compute FAO-56 Penman-Monteith hourly reference evapotranspiration.

    Implements the hourly reference ET equation from FAO Irrigation and
    Drainage Paper No. 56 (Allen et al., 1998), Section 4.2.

    Parameters
    ----------
    temp_k : xr.DataArray or np.ndarray
        Air temperature at 2 m in Kelvin.
    dewpoint_k : xr.DataArray or np.ndarray
        Dewpoint temperature in Kelvin.
    wind_speed_2m : xr.DataArray or np.ndarray
        Wind speed at 2 m height in m/s.
    solar_rad : xr.DataArray or np.ndarray
        Incoming shortwave solar radiation in W/m2.
    pressure : xr.DataArray or np.ndarray
        Atmospheric pressure in Pa.
    lat : xr.DataArray, np.ndarray, or None
        Latitude in degrees (reserved for future net longwave radiation computation).
    doy : xr.DataArray, np.ndarray, or None
        Day of year (reserved).

    Returns
    -------
    xr.DataArray or np.ndarray
        Reference ET in mm/hour.  The return type mirrors the input type:
        if *temp_k* is an :class:`xr.DataArray` the result carries
        ``name="pet"`` and ``attrs["units"] = "mm/hr"``.

    Notes
    -----
    The FAO-56 hourly reference ET is::

        ET0 = (0.408 * delta * (Rn - G) + gamma * (37 / (T + 273)) * u2 * (es - ea))
              / (delta + gamma * (1 + 0.34 * u2))

    where delta is the slope of the saturation vapour pressure curve (kPa/degC),
    Rn is net radiation (MJ/m2/hr), G is soil heat flux (set to 0),
    gamma is the psychrometric constant (kPa/degC), T is air temperature (degC),
    u2 is wind speed at 2 m (m/s), es is saturation vapour pressure (kPa),
    and ea is actual vapour pressure (kPa).
    """
    # Convert temperatures to Celsius
    T = temp_k - 273.15  # noqa: N806
    Td = dewpoint_k - 273.15  # noqa: N806

    # Psychrometric constant gamma = 0.000665 * P  (kPa/degC, P in kPa)
    P_kpa = pressure / 1000.0  # Pa to kPa  # noqa: N806
    gamma = 0.000665 * P_kpa

    # Saturation vapour pressure es (kPa) at air temperature
    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))

    # Actual vapour pressure ea (kPa) from dewpoint
    ea = 0.6108 * np.exp(17.27 * Td / (Td + 237.3))

    # Slope of saturation vapour pressure curve delta (kPa/degC)
    delta = 4098.0 * es / (T + 237.3) ** 2

    # Net radiation: approximate Rn = 0.77 * Rs (albedo 0.23, ignoring longwave)
    # Convert W/m2 to MJ/m2/hr  (1 W/m2 = 0.0036 MJ/m2/hr)
    Rn = 0.77 * solar_rad * 0.0036  # MJ/m2/hr  # noqa: N806

    # Soil heat flux G = 0 (simplification)
    G = 0.0  # noqa: N806

    # FAO-56 hourly PM equation
    numerator = 0.408 * delta * (Rn - G) + gamma * (37.0 / (T + 273.0)) * wind_speed_2m * (
        es - ea
    )
    denominator = delta + gamma * (1.0 + 0.34 * wind_speed_2m)

    et0 = numerator / denominator

    # Clip to non-negative values
    et0 = np.maximum(et0, 0.0)

    if isinstance(et0, xr.DataArray):
        et0.name = "pet"
        et0.attrs["units"] = "mm/hr"
        et0.attrs["long_name"] = "FAO-56 Penman-Monteith reference ET"
    return et0
