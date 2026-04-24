from __future__ import annotations
import numpy as np
import xarray as xr

from __future__ import annotations
import numpy as np
import xarray as xr

def adjust_wind_height(
    wind_speed: xr.DataArray,
    z_from: float = 10.0,
    z_to: float = 2.0,
    method: str = "logarithmic",
    alpha: float = 1 / 7,
) -> xr.DataArray:
    """Adjust wind speed between measurement heights."""
    # Local coefficients for FAO-56 Eq 47 logarithmic profile assumption (short grass)


    if method == "logarithmic":
        FAO_MULTIPLIER = 67.8
        FAO_OFFSET = 5.42
        factor = np.log(FAO_MULTIPLIER * z_to - FAO_OFFSET) / np.log(FAO_MULTIPLIER * z_from - FAO_OFFSET)
        adjusted = wind_speed * factor
    elif method == "power_law":
        adjusted = wind_speed * (z_to / z_from) ** alpha
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'logarithmic' or 'power_law'.")

    return adjusted.rename("wind_speed_ms")

def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Compute wind speed from U and V components using MetPy."""
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
