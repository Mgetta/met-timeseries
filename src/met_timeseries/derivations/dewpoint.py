"""Dewpoint temperature derivation from specific humidity, pressure, and temperature."""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_dewpoint(
    specific_humidity: xr.DataArray | np.ndarray,
    pressure: xr.DataArray | np.ndarray,
    temperature: xr.DataArray | np.ndarray,
) -> xr.DataArray | np.ndarray:
    """Compute dewpoint temperature from specific humidity, pressure, and temperature.

    Uses the Magnus-Tetens approximation.  The actual vapour pressure is
    derived from specific humidity and surface pressure, then inverted to
    give dewpoint.

    Parameters
    ----------
    specific_humidity : xr.DataArray or np.ndarray
        Specific humidity in kg/kg.
    pressure : xr.DataArray or np.ndarray
        Surface pressure in Pa.
    temperature : xr.DataArray or np.ndarray
        Air temperature in K (used only to clip physically unrealistic
        dewpoints above the air temperature).

    Returns
    -------
    xr.DataArray or np.ndarray
        Dewpoint temperature in K.  If the input is an
        :class:`xr.DataArray` the result carries ``name="dewpoint"`` and
        ``attrs["units"] = "K"``.

    Notes
    -----
    The actual vapour pressure is calculated as::

        e = (q * P) / (0.622 + 0.378 * q)

    where *q* is specific humidity (kg/kg) and *P* is pressure (Pa).
    The result is converted to hPa before applying the Magnus formula::

        Td = 243.5 * ln(e / 6.112) / (17.67 - ln(e / 6.112)) + 273.15

    Examples
    --------
    >>> import numpy as np
    >>> td = compute_dewpoint(
    ...     np.array([0.01]),
    ...     np.array([101325.0]),
    ...     np.array([300.0]),
    ... )
    >>> float(td)  # doctest: +ELLIPSIS
    284...
    """
    q = specific_humidity
    P = pressure  # noqa: N806 (conventional meteorological name)

    # Actual vapour pressure in Pa
    e_pa = (q * P) / (0.622 + 0.378 * q)

    # Convert Pa to hPa
    e_hpa = e_pa / 100.0

    # Clip to avoid log(0) or negative values
    e_hpa = np.maximum(e_hpa, 1e-10)

    ln_e = np.log(e_hpa / 6.112)
    td_celsius = 243.5 * ln_e / (17.67 - ln_e)
    td_kelvin = td_celsius + 273.15

    # Dewpoint cannot exceed air temperature
    td_kelvin = np.minimum(td_kelvin, temperature)

    if isinstance(td_kelvin, xr.DataArray):
        td_kelvin.name = "dewpoint"
        td_kelvin.attrs["units"] = "K"
        td_kelvin.attrs["long_name"] = "Dewpoint temperature"
    return td_kelvin
