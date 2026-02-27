"""Wind speed derivation from horizontal u and v components."""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_wind_speed(
    u: xr.DataArray | np.ndarray,
    v: xr.DataArray | np.ndarray,
) -> xr.DataArray | np.ndarray:
    """Compute wind speed magnitude from eastward and northward components.

    Parameters
    ----------
    u : xr.DataArray or np.ndarray
        Eastward (zonal) wind component in m/s.
    v : xr.DataArray or np.ndarray
        Northward (meridional) wind component in m/s.

    Returns
    -------
    xr.DataArray or np.ndarray
        Wind speed ``sqrt(u^2 + v^2)`` in m/s.  The return type mirrors the
        input type: if *u* is an :class:`xr.DataArray` the result is also
        an :class:`xr.DataArray` with ``name="wind_speed"`` and
        ``attrs["units"] = "m/s"``; otherwise a plain :class:`np.ndarray`
        is returned.

    Examples
    --------
    >>> import numpy as np
    >>> compute_wind_speed(np.array([3.0]), np.array([4.0]))
    array([5.])
    """
    result = np.sqrt(u**2 + v**2)
    if isinstance(result, xr.DataArray):
        result.name = "wind_speed"
        result.attrs["units"] = "m/s"
        result.attrs["long_name"] = "Wind speed"
    return result
