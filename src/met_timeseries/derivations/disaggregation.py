"""Temporal disaggregation of PRISM daily precipitation to hourly."""

from __future__ import annotations

import numpy as np
import xarray as xr


def disaggregate_precip(
    prism_daily: xr.DataArray,
    nldas_hourly: xr.DataArray,
) -> xr.DataArray:
    """Disaggregate PRISM daily precipitation to hourly using NLDAS fractions.

    Scales each PRISM daily total proportionally to the NLDAS hourly
    precipitation pattern for the same day.  This preserves the spatial
    distribution of PRISM while using NLDAS for the sub-daily temporal
    pattern.

    Steps:

    1. For each grid cell, compute the NLDAS hourly fraction of the daily
       total (``nldas_hour / nldas_daily_sum``).
    2. Multiply the PRISM daily total by these fractions.
    3. Where the NLDAS daily total is zero, distribute precipitation
       uniformly across all hours (``prism_daily / 24``).

    Parameters
    ----------
    prism_daily : xr.DataArray
        PRISM daily precipitation totals in mm/day with dimensions
        ``(time, lat, lon)``.  The ``time`` coordinate must have daily
        frequency.
    nldas_hourly : xr.DataArray
        NLDAS-2 hourly precipitation in mm/hr (or any consistent unit)
        with dimensions ``(time, y, x)`` or ``(time, lat, lon)``.  The
        ``time`` coordinate must have hourly frequency and cover the
        same calendar days as *prism_daily*.

    Returns
    -------
    xr.DataArray
        Hourly precipitation in mm/hr with the same spatial grid as
        *prism_daily* and the same temporal resolution as *nldas_hourly*.
        Carries ``name="prcp"`` and ``attrs["units"] = "mm/hr"``.

    Notes
    -----
    The two datasets may have different spatial grids.  *nldas_hourly* is
    used **only** for its temporal pattern; the spatial structure comes
    from *prism_daily*.  The NLDAS pattern is spatially averaged over the
    domain before computing hourly fractions.
    """
    # Spatial average of NLDAS hourly precip (domain mean temporal pattern)
    spatial_dims = [d for d in nldas_hourly.dims if d != "time"]
    nldas_mean = nldas_hourly.mean(dim=spatial_dims)  # (time_hourly,)

    # Group hourly NLDAS by calendar day -> daily sums
    nldas_daily_sum = nldas_mean.resample(time="1D").sum()  # (time_daily,)

    # Build hourly fractions: for each hour, fraction of its day's total
    hourly_dates = nldas_mean.time.dt.floor("D")
    hourly_fractions = xr.DataArray(
        np.zeros(len(nldas_mean), dtype=np.float32),
        coords={"time": nldas_mean.time},
        dims=["time"],
    )

    unique_days = np.unique(hourly_dates.values)
    for day in unique_days:
        day_mask = hourly_dates.values == day
        daily_total = float(nldas_daily_sum.sel(time=day).values)
        if daily_total > 0:
            hourly_fractions.values[day_mask] = (
                nldas_mean.values[day_mask] / daily_total
            )
        else:
            # Uniform distribution
            n_hours = day_mask.sum()
            hourly_fractions.values[day_mask] = 1.0 / n_hours

    # Expand prism_daily to hourly by broadcasting with the fractions
    prism_time = prism_daily.time.values  # daily timestamps
    prism_time_floor = np.array(prism_time, dtype="datetime64[D]")

    hourly_times = nldas_mean.time.values
    hourly_dates_vals = np.array(hourly_dates.values, dtype="datetime64[D]")

    # Look up PRISM day for each hourly step
    day_to_prism = {day: val for day, val in zip(prism_time_floor, prism_daily.values)}

    lat_size = prism_daily.sizes.get("lat", prism_daily.sizes.get("y", 1))
    lon_size = prism_daily.sizes.get("lon", prism_daily.sizes.get("x", 1))

    output = np.zeros((len(hourly_times), lat_size, lon_size), dtype=np.float32)
    for i, (ht, hd) in enumerate(zip(hourly_times, hourly_dates_vals)):
        hd_dt64 = np.datetime64(hd, "D")
        if hd_dt64 in day_to_prism:
            output[i] = day_to_prism[hd_dt64] * float(hourly_fractions.values[i])

    lat_coord = "lat" if "lat" in prism_daily.coords else "y"
    lon_coord = "lon" if "lon" in prism_daily.coords else "x"

    result = xr.DataArray(
        output,
        coords={
            "time": hourly_times,
            lat_coord: prism_daily[lat_coord].values,
            lon_coord: prism_daily[lon_coord].values,
        },
        dims=["time", lat_coord, lon_coord],
        name="prcp",
        attrs={"units": "mm/hr", "long_name": "Disaggregated hourly precipitation"},
    )
    return result
