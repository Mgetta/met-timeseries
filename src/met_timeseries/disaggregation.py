"""
Source-agnostic temporal disaggregation.

Core API
--------
disaggregate(coarse, conservation, ...)
    Generic disaggregation. Conservation property determines the math:
    - "sum"                 : extensive variables (precipitation, radiation)
    - "mean_additive"       : intensive additive (temperature, dewpoint)
    - "mean_multiplicative" : intensive multiplicative (wind speed)

    For sum-conserving disaggregation, the weight source is controlled by
    weight_method:
    - "proportional" : observed fine_pattern used as weight shape
    - "trapezoidal"  : synthetic solar geometry trapezoid (no fine data needed)
    - "solar"        : pvlib clearsky radiation shape (no fine data needed)
    - "custom"       : caller supplies pre-computed normalised weights

disaggregate_precipitation_stochastic(coarse, fine_pattern, method)
    Stochastic cascade disaggregation for precipitation.
    Separate from the generic framework as it is non-deterministic.

Convenience wrappers
--------------------
disaggregate_precipitation, disaggregate_radiation, disaggregate_temperature,
disaggregate_dewpoint, disaggregate_wind, disaggregate_pet_trapezoidal,
disaggregate_pet_solar

All operate on xr.DataArray with a 'time' coordinate.
Frequencies are inferred automatically — no hardcoded "1h" or "1D".

Legacy pandas API
-----------------
The original pandas-based functions are preserved for backward compatibility:
disaggregate_temperature_pattern, disaggregate_temperature_sine,
disaggregate_radiation_pattern, disaggregate_radiation_potential,
disaggregate_wind_pattern, disaggregate_wind_equal,
disaggregate_dewpoint_pattern, disaggregate_dewpoint_constant.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import poisson
from mettoolbox.melodist.melodist.precipitation import build_casc, disagg_prec

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ConservationMethod = Literal[
    "sum",                   # extensive: precipitation, radiation totals
    "mean_additive",         # intensive additive: temperature, dewpoint
    "mean_multiplicative",   # intensive multiplicative: wind speed
]

WeightMethod = Literal["proportional", "trapezoidal", "solar", "custom"]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _proportional_disaggregate(
    daily_total: pd.Series,
    hourly_pattern: pd.Series,
    *,
    name: str,
) -> pd.Series:
    """Distribute a daily total proportionally based on an hourly pattern.

    For each day:
    - If the day's pattern sum is positive: ``hourly = daily_total × (hour / day_sum)``.
    - If the day's pattern sums to zero: spread equally across hours in the day.

    Parameters
    ----------
    daily_total:
        Daily totals with a daily :class:`~pandas.DatetimeIndex`.
    hourly_pattern:
        Hourly template values with an hourly :class:`~pandas.DatetimeIndex`.
    name:
        Name to assign to the output :class:`~pandas.Series`.

    Returns
    -------
    :class:`pandas.Series` with the same index as *hourly_pattern*, named *name*.
    """
    result = pd.Series(np.nan, index=hourly_pattern.index, dtype=float)

    timestamps = daily_total.index
    for idx, (date, daily_value) in enumerate(daily_total.items()):
        start = pd.Timestamp(date)
        if idx + 1 < len(timestamps):
            end = pd.Timestamp(timestamps[idx + 1])
        else:
            end = start + pd.Timedelta(hours=24)

        day_mask = (hourly_pattern.index >= start) & (hourly_pattern.index < end)
        day_hours = hourly_pattern[day_mask]

        if len(day_hours) == 0:
            continue

        day_sum = day_hours.sum()
        if day_sum > 0:
            fractions = day_hours / day_sum
        else:
            fractions = pd.Series(1.0 / len(day_hours), index=day_hours.index)

        result[day_mask] = daily_value * fractions

    result.name = name
    return result


# ---------------------------------------------------------------------------
# Precipitation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cascade internal helpers
# ---------------------------------------------------------------------------

def _fit_molnar_burlando(precip: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit log-Poisson parameters and return (A, beta, tau_obs)."""

    n = len(precip)
    precip = precip[: n - n % 32]

    scales = [precip]
    for _ in range(5):
        scales.append(scales[-1].reshape(-1, 2).sum(axis=1))

    q = np.arange(0.5, 5.5, 0.5)
    log_lambda = np.log([32, 16, 8, 4, 2, 1])
    moments = np.zeros((6, len(q)))
    for s in range(6):
        for i, qi in enumerate(q):
            moments[s, i] = np.mean(scales[s] ** qi)

    tau_obs = np.array([
        -np.polyfit(log_lambda, np.log(moments[:, i]), 1)[0]
        for i in range(len(q))
    ])

    def objective(params):
        c, beta = abs(params[0]), abs(params[1])
        tau_pred = q - c * (q * (1 - beta) + beta**q - 1) / np.log(2)
        return np.sqrt(np.mean((tau_pred - tau_obs) ** 2))

    result = minimize(objective, x0=[0.4, 0.2], method="Nelder-Mead")
    c = abs(result.x[0])
    beta = abs(result.x[1])
    A = np.exp(c * (1 - beta))

    return A, beta, tau_obs



def _molnar_burlando_split(values, A, beta, tau_obs):
    n = len(values)
    out = np.zeros(n * 2)

    prob_dry = 1.0 - 2 ** (-1.0 + tau_obs[0])
    prob_dry = prob_dry * (1 + prob_dry)

    for i in range(n):
        if values[i] == 0:
            continue

        W = A * (beta ** poisson.rvs(1, size=2))
        W[W < 0] = 0

        wet = np.random.binomial(1, 1 - prob_dry, size=2)
        if wet[0] == 0 and wet[1] == 0:
            wet[np.random.randint(2)] = 1

        W = W * wet
        w_sum = W[0] + W[1]
        if w_sum <= 0:
            W[:] = 0.5
            w_sum = 1.0

        out[2 * i] = values[i] * W[0] / w_sum
        out[2 * i + 1] = values[i] * W[1] / w_sum

    return out


def _cascade_to_hourly(values: np.ndarray, daily_total: pd.Series) -> pd.Series:
    """Convert 45-min cascade output to hourly via Güntner et al. (2001).

    45-min values are uniformly split into three 15-min values, then
    summed in groups of four to produce 60-min totals.
    """
    fifteen_min = np.repeat(values / 3.0, 3)
    fifteen_min = fifteen_min[: (len(fifteen_min) // 4) * 4]
    hourly = fifteen_min.reshape(-1, 4).sum(axis=1)

    n_hours = len(daily_total) * 24
    hourly = hourly[:n_hours]

    hourly_index = pd.date_range(
        start=daily_total.index[0],
        periods=n_hours,
        freq="h",
    )

    result = pd.Series(hourly, index=hourly_index, dtype=float)
    result.name = "precip_mm"
    return result


def _molnar_burlando_disagg(daily_total, hourly_pattern, n_levels=5):
    A, beta, tau_obs = _fit_molnar_burlando(hourly_pattern.values)
    values = daily_total.values.copy()

    for _ in range(n_levels):
        values = _molnar_burlando_split(values, A, beta, tau_obs)

    return _cascade_to_hourly(values, daily_total)


def _olsson_disagg(daily_total, hourly_pattern):

    hourly_df = pd.DataFrame({"precip": hourly_pattern})
    stats = build_casc(hourly_df, months=[np.arange(12) + 1])[0]
    daily_df = pd.DataFrame({"precip": daily_total})
    return disagg_prec(daily_df, method="cascade", cascade_options=stats)


def _disaggregate_precipitation_hybrid(
    daily_total: pd.Series,
    hourly_pattern: pd.Series,
    cascade_method: str = "molnar_burlando",
) -> pd.Series:
    """Disaggregate daily precipitation using PRISM/NLDAS hybrid logic.

    Per-day rules:
    - PRISM > 0 and NLDAS > 0: proportional disaggregation
    - PRISM == 0 and NLDAS > 0: use NLDAS hourly values directly
    - PRISM > 0 and NLDAS == 0: cascade disaggregation

    Parameters
    ----------
    daily_total:
        Daily precipitation totals (e.g. PRISM) in mm with a daily
        :class:`~pandas.DatetimeIndex`.
    hourly_pattern:
        Hourly precipitation (e.g. NLDAS) with an hourly
        :class:`~pandas.DatetimeIndex`.
    cascade_method:
        Cascade method to use when NLDAS is zero. One of
        ``"molnar_burlando"`` or ``"olsson"``.

    Returns
    -------
    :class:`pandas.Series` named ``precip_mm``.
    """
    proportional = disaggregate_precipitation(daily_total, hourly_pattern, method="proportional")
    cascade = disaggregate_precipitation(daily_total, hourly_pattern, method=cascade_method)

    result = pd.Series(np.nan, index=hourly_pattern.index, dtype=float)

    cascade_days = []
    for timestamp, prism_total in daily_total.items():
        day_mask = pd.date_range(start=timestamp, end=timestamp + pd.Timedelta(hours=23),freq = 'h')

        nldas_sum = hourly_pattern[hourly_pattern.index.intersection(day_mask)].sum()
    
        if prism_total > 0 and nldas_sum > 0:
            day_mask = proportional.index.intersection(day_mask)
            result[day_mask] = proportional[day_mask]
        elif prism_total == 0 and nldas_sum > 0:
            day_mask = hourly_pattern.index.intersection(day_mask)
            result[day_mask] = hourly_pattern[day_mask]
        elif prism_total > 0 and nldas_sum == 0:
            day_mask = cascade.index.intersection(day_mask)
            result[day_mask] = cascade[day_mask]
            cascade_days.append(timestamp)
        elif all(day_mask.isin(hourly_pattern.index)) and all(day_mask.isin(proportional.index)):
            # both zero — no rain
            result[day_mask] = 0.0
        else:
            print(f"Missing data case for timestamp {timestamp}: leave as NaN")
            pass

    result.name = "precip_mm"
    return result, cascade_days
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
  

def disaggregate_precipitation(
    daily_total_or_coarse: pd.Series | xr.DataArray,
    hourly_pattern_or_fine: pd.Series | xr.DataArray,
    method: str = "molnar_burlando",
) -> pd.Series | xr.DataArray:
    """Disaggregate daily precipitation to finer resolution.

    Dispatches to the xarray API when passed :class:`xr.DataArray` inputs
    (sum-conserving proportional disaggregation), or to the legacy pandas
    cascade path when passed :class:`pandas.Series` inputs.

    Parameters
    ----------
    daily_total_or_coarse:
        Daily precipitation totals (pd.Series) or a coarse-resolution
        xr.DataArray with a ``time`` coordinate.
    hourly_pattern_or_fine:
        Hourly precipitation pattern (pd.Series) or a fine-resolution
        xr.DataArray used as the proportional weight shape.
    method:
        Only used for the legacy pandas path.
        ``"molnar_burlando"`` (default), ``"olsson"``, ``"proportional"``,
        or ``"hybrid"``.

    Returns
    -------
    :class:`pandas.Series` named ``precip_mm`` (pandas path) or
    :class:`xr.DataArray` (xarray path).
    """
    if isinstance(daily_total_or_coarse, xr.DataArray):
        return disaggregate(
            daily_total_or_coarse,
            conservation="sum",
            fine_pattern=hourly_pattern_or_fine,
        )
    # Legacy pandas path
    daily_total = daily_total_or_coarse
    hourly_pattern = hourly_pattern_or_fine
    if method == "molnar_burlando":
        return _molnar_burlando_disagg(daily_total, hourly_pattern)
    elif method == "olsson":
        return _olsson_disagg(daily_total, hourly_pattern)
    elif method == "proportional":
        return _proportional_disaggregate(daily_total, hourly_pattern, name="precip_mm")
    elif method == "hybrid":
        return _disaggregate_precipitation_hybrid(daily_total, hourly_pattern)
    else:
        raise ValueError(f"Unknown cascade method: {method!r}")
# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


def disaggregate_temperature_sine(
    daily_tmin: pd.Series,
    daily_tmax: pd.Series,
    method: str = "sine_min_max",
    daily_tmean: pd.Series | None = None,
) -> pd.Series:
    """Disaggregate daily min/max temperature to hourly using a sine curve.

    Wraps ``mettoolbox.melodist.melodist.temperature.disaggregate_temperature``
    with fixed sun times (sunrise=7 h, noon=12 h, sunset=19 h).

    Parameters
    ----------
    daily_tmin:
        Daily minimum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    daily_tmax:
        Daily maximum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    method:
        Disaggregation method passed to MELODIST (default ``"sine_min_max"``).
    daily_tmean:
        Optional daily mean temperature in °C.  When ``None`` the mean is
        computed as ``(tmin + tmax) / 2``.

    Returns
    -------
    :class:`pandas.Series` named ``temp_c``.
    """
    from mettoolbox.melodist.melodist.temperature import disaggregate_temperature

    tmean = daily_tmean if daily_tmean is not None else (daily_tmin + daily_tmax) / 2.0

    data_daily = pd.DataFrame(
        {
            "tmin": daily_tmin.values,
            "tmax": daily_tmax.values,
            "temp": tmean.values,
        },
        index=daily_tmin.index,
    )

    result = disaggregate_temperature(data_daily, method=method, min_max_time="fix")
    result.name = "temp_c"
    return result


def disaggregate_temperature_pattern(
    daily_tmin: pd.Series,
    daily_tmax: pd.Series,
    hourly_pattern: pd.Series,
) -> pd.Series:
    """Disaggregate daily temperature using linear rescaling of an hourly pattern.

    For each day the hourly pattern is linearly rescaled from its
    ``[pattern_min, pattern_max]`` to the target ``[daily_tmin, daily_tmax]``.
    Days where the pattern is flat (``pattern_max == pattern_min``) are filled
    with ``(tmin + tmax) / 2``.

    Parameters
    ----------
    daily_tmin:
        Daily minimum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    daily_tmax:
        Daily maximum temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    hourly_pattern:
        Hourly temperature pattern with an hourly :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    :class:`pandas.Series` named ``temp_c``.
    """
    result = pd.Series(np.nan, index=hourly_pattern.index, dtype=float)

    for date in daily_tmin.index:
        date_key = pd.Timestamp(date).date()
        day_mask = hourly_pattern.index.date == date_key
        day_hours = hourly_pattern[day_mask]

        if len(day_hours) == 0:
            continue

        tmin_val = float(daily_tmin[date])
        tmax_val = float(daily_tmax[date])

        p_min = float(day_hours.min())
        p_max = float(day_hours.max())

        if p_max == p_min:
            result[day_mask] = (tmin_val + tmax_val) / 2.0
        else:
            rescaled = (day_hours - p_min) / (p_max - p_min) * (tmax_val - tmin_val) + tmin_val
            result[day_mask] = rescaled

    result.name = "temp_c"
    return result


# ---------------------------------------------------------------------------
# Radiation
# ---------------------------------------------------------------------------


def disaggregate_radiation_pattern(
    daily_total: pd.Series,
    hourly_pattern: pd.Series,
) -> pd.Series:
    """Distribute daily shortwave radiation proportionally using an hourly pattern.

    Parameters
    ----------
    daily_total:
        Daily total shortwave radiation in W/m² (or MJ/m²/day) with a daily
        :class:`~pandas.DatetimeIndex`.
    hourly_pattern:
        Hourly shortwave radiation pattern with an hourly
        :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    :class:`pandas.Series` named ``shortwave_wm2``.
    """
    return _proportional_disaggregate(daily_total, hourly_pattern, name="shortwave_wm2")


def disaggregate_radiation_potential(
    daily_total: pd.Series,
    lat: float,
    lon: float,
) -> pd.Series:
    """Distribute daily shortwave radiation using a clear-sky curve as temporal template.

    When no observed hourly pattern is available, theoretical clear-sky radiation
    provides the diurnal shape.  The curve is scaled so that each day's sum
    matches *daily_total*.

    Parameters
    ----------
    daily_total:
        Daily total shortwave radiation with a daily
        :class:`~pandas.DatetimeIndex`.
    lat:
        Latitude in decimal degrees.
    lon:
        Longitude in decimal degrees.

    Returns
    -------
    :class:`pandas.Series` named ``shortwave_wm2``.
    """
    from met_timeseries.derivations import _clear_sky_radiation, _to_datetime

    hourly_index = pd.date_range(
        start=daily_total.index[0],
        end=daily_total.index[-1] + pd.Timedelta(hours=23),
        freq="h",
    )

    clearsky = pd.Series(
        [_clear_sky_radiation(lat, lon, _to_datetime(t)) for t in hourly_index],
        index=hourly_index,
        dtype=float,
    )

    return _proportional_disaggregate(daily_total, clearsky, name="shortwave_wm2")


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------


def disaggregate_wind_pattern(
    daily_mean: pd.Series,
    hourly_pattern: pd.Series,
) -> pd.Series:
    """Scale hourly wind pattern to match a daily mean target.

    Multiplicative scaling: ``hourly × (target_daily_mean / pattern_daily_mean)``.
    Days where the pattern mean is zero are filled with the target daily mean.
    Output is clipped to ``>= 0``.

    Parameters
    ----------
    daily_mean:
        Target daily mean wind speed in m/s with a daily
        :class:`~pandas.DatetimeIndex`.
    hourly_pattern:
        Hourly wind speed pattern with an hourly :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    :class:`pandas.Series` named ``wind_speed_ms``.
    """
    result = pd.Series(np.nan, index=hourly_pattern.index, dtype=float)

    for date, target_mean in daily_mean.items():
        date_key = pd.Timestamp(date).date()
        day_mask = hourly_pattern.index.date == date_key
        day_hours = hourly_pattern[day_mask]

        if len(day_hours) == 0:
            continue

        pattern_mean = float(day_hours.mean())
        if pattern_mean > 0:
            result[day_mask] = day_hours * (float(target_mean) / pattern_mean)
        else:
            result[day_mask] = float(target_mean)

    result = result.clip(lower=0.0)
    result.name = "wind_speed_ms"
    return result


def disaggregate_wind_equal(daily_mean: pd.Series) -> pd.Series:
    """Repeat daily mean wind speed as a constant across each day.

    Parameters
    ----------
    daily_mean:
        Daily mean wind speed in m/s with a daily :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    :class:`pandas.Series` with an hourly index, named ``wind_speed_ms``.
    """
    hourly_index = pd.date_range(
        start=daily_mean.index[0],
        end=daily_mean.index[-1] + pd.Timedelta(hours=23),
        freq="h",
    )
    result = daily_mean.reindex(hourly_index, method="ffill")
    result.name = "wind_speed_ms"
    return result


# ---------------------------------------------------------------------------
# Dewpoint
# ---------------------------------------------------------------------------


def disaggregate_dewpoint_pattern(
    daily_dewpoint: pd.Series,
    hourly_pattern: pd.Series,
) -> pd.Series:
    """Bias-correct an hourly dewpoint pattern to match a daily target.

    Additive bias correction:
    ``hourly_output = hourly_pattern + (daily_target - pattern_daily_mean)``

    Parameters
    ----------
    daily_dewpoint:
        Target daily mean dewpoint in °C with a daily
        :class:`~pandas.DatetimeIndex`.
    hourly_pattern:
        Hourly dewpoint pattern with an hourly :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    :class:`pandas.Series` named ``dewpoint_c``.
    """
    result = pd.Series(np.nan, index=hourly_pattern.index, dtype=float)

    for date, target in daily_dewpoint.items():
        date_key = pd.Timestamp(date).date()
        day_mask = hourly_pattern.index.date == date_key
        day_hours = hourly_pattern[day_mask]

        if len(day_hours) == 0:
            continue

        pattern_mean = float(day_hours.mean())
        bias = float(target) - pattern_mean
        result[day_mask] = day_hours + bias

    result.name = "dewpoint_c"
    return result


def disaggregate_dewpoint_constant(daily_dewpoint: pd.Series) -> pd.Series:
    """Repeat daily dewpoint as a constant across each day.

    Parameters
    ----------
    daily_dewpoint:
        Daily dewpoint temperature in °C with a daily
        :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    :class:`pandas.Series` with an hourly index, named ``dewpoint_c``.
    """
    hourly_index = pd.date_range(
        start=daily_dewpoint.index[0],
        end=daily_dewpoint.index[-1] + pd.Timedelta(hours=23),
        freq="h",
    )
    result = daily_dewpoint.reindex(hourly_index, method="ffill")
    result.name = "dewpoint_c"
    return result


# ---------------------------------------------------------------------------
# PET
# ---------------------------------------------------------------------------


def disaggregate_pevt(pevt_daily: xr.DataArray, method: str = "diurnal") -> xr.DataArray:
    """
    Downscale daily PEVT into hourly values.

    Parameters:
    - pevt_daily (xr.DataArray): Daily Potential Evapotranspiration (PEVT) values.
        Must have 'time' coordinates in daily resolution with units (mm/day).
    - method (str): Method used for disaggregation. Options are:
        - "diurnal" (default): Weight based on diurnal variability curve.
        - "static": Uniform hourly distribution.

    Returns:
    - xr.DataArray: Hourly PEVT values (mm/hour), aligned with expanded coordinates.

    Notes:
    - Input is assumed to be daily resolution with units of mm/day.
    - Output is hourly resolution with units converted to mm/hour.
    - The diurnal method uses weights informed by typical evapotranspiration patterns.
    
    References:
    - Allen et al., 1998. Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements (FAO-56).
    """
    if "time" not in pevt_daily.coords:
        raise ValueError("Input DataArray must contain 'time' coordinates.")

    if pevt_daily.resample(time="1D").count().size != pevt_daily.time.size:
        raise ValueError("Input data should be daily; check the time resolution.")

    if method == "static":
        # Static disaggregation: evenly spread mm/day across 24 hours
        hourly_values = pevt_daily / 24.0
        pevt_hourly = hourly_values.resample(time="1H").ffill()

    elif method == "diurnal":
        # Diurnal Dynamic Disaggregation
        diurnal_weights = np.array([
            0.05, 0.05, 0.03, 0.03, 0.02, 0.03,  # Midnight-06:00
            0.05, 0.10, 0.15, 0.20, 0.15, 0.10,  # 06:00-Noon
            0.08, 0.07, 0.05, 0.03, 0.02, 0.03,  # Noon-18:00
            0.03, 0.05, 0.07, 0.08, 0.05, 0.05   # 18:00-Midnight
        ])
        diurnal_weights /= diurnal_weights.sum()  # Normalize weights
        scale_factor = pevt_daily / diurnal_weights.sum()

        # Expand weights across time range
        pevt_hourly = (
            pevt_daily.resample(time="1H").ffill() * scale_factor
        )

    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'static' or 'diurnal'.")

    # Add metadata
    pevt_hourly.attrs["units"] = "mm/hour"
    pevt_hourly.attrs["description"] = (
        f"Potential Evapotranspiration (PEVT) downscaled to hourly using"
        f"the '{method}' method."
    )

    return pevt_hourly

# ===========================================================================
# xarray-native, frequency-agnostic disaggregation framework
# ===========================================================================

# ---------------------------------------------------------------------------
# Frequency inference helpers
# ---------------------------------------------------------------------------


def _infer_freq(da: xr.DataArray) -> str:
    """Infer time frequency string from a DataArray's time coordinate."""
    times = pd.DatetimeIndex(da.coords["time"].values)
    freq = pd.infer_freq(times)
    if freq is None:
        name_hint = f" '{da.name}'" if da.name else ""
        time_hint = f" (first timestamp: {times[0]})" if len(times) else ""
        raise ValueError(
            f"Cannot infer time frequency for DataArray{name_hint}{time_hint}"
            " — check for gaps or irregular spacing"
        )
    return freq


def _count_fine_per_coarse(fine: xr.DataArray, coarse_freq: str) -> xr.DataArray:
    """Count fine timesteps per coarse period, broadcast back to fine resolution."""
    counts = xr.ones_like(fine).resample(time=coarse_freq).sum()
    return counts.reindex_like(fine, method="ffill")


# ---------------------------------------------------------------------------
# Private core implementations (xarray-native, vectorized)
# ---------------------------------------------------------------------------


def _disaggregate_sum(coarse: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """Sum-conserving disaggregation: fine values sum back to the coarse total."""
    coarse_h = coarse.reindex_like(weights, method="ffill")
    return (coarse_h * weights).clip(min=0.0)


def _disaggregate_mean_additive(
    coarse: xr.DataArray, fine_pattern: xr.DataArray
) -> xr.DataArray:
    """Additive bias correction: shift fine pattern so its mean matches coarse value.

    Preserves absolute differences. Best for temperature, dewpoint.
    """
    coarse_freq = _infer_freq(coarse)
    pattern_mean = fine_pattern.resample(time=coarse_freq).mean()
    bias = coarse - pattern_mean
    bias_h = bias.reindex_like(fine_pattern, method="ffill")
    return fine_pattern + bias_h


def _disaggregate_mean_multiplicative(
    coarse: xr.DataArray, fine_pattern: xr.DataArray
) -> xr.DataArray:
    """Multiplicative scaling: scale fine pattern so its mean matches coarse value.

    Preserves relative variability. Best for wind speed.
    Falls back to coarse value where pattern mean is zero.
    """
    coarse_freq = _infer_freq(coarse)
    pattern_mean = fine_pattern.resample(time=coarse_freq).mean()
    scale = xr.where(
        pattern_mean > 0,
        coarse / pattern_mean,
        # np.nan used as sentinel: these cells fall back to the coarse value
        # (handled below via isnull check) rather than the scaled fine pattern.
        np.nan,
    )
    scale_h = scale.reindex_like(fine_pattern, method="ffill")
    coarse_h = coarse.reindex_like(fine_pattern, method="ffill")
    return xr.where(scale_h.isnull(), coarse_h, (fine_pattern * scale_h).clip(min=0.0))


# ---------------------------------------------------------------------------
# Weight generators
# ---------------------------------------------------------------------------


def _normalise_weights(fine_pattern: xr.DataArray, coarse_freq: str) -> xr.DataArray:
    """Normalise a fine-resolution array so it sums to 1.0 per coarse period.

    Uniform fallback (1/n_fine) where period sum is zero.
    """
    period_sum = fine_pattern.resample(time=coarse_freq).sum()
    # Replace zeros with 1.0 for safe division; track original zeros separately
    safe_sum = period_sum.where(period_sum > 0, other=1.0)
    safe_sum_h = safe_sum.reindex_like(fine_pattern, method="ffill")
    period_sum_h = period_sum.reindex_like(fine_pattern, method="ffill")
    n_fine = _count_fine_per_coarse(fine_pattern, coarse_freq)
    return xr.where(period_sum_h > 0, fine_pattern / safe_sum_h, 1.0 / n_fine)


def _trapezoidal_weights(
    template: xr.DataArray,
    fine_freq: str = "1h",
    coarse_freq: str = "1D",
) -> xr.DataArray:
    """Generate normalised weights from solar trapezoid geometry.

    Refactored from the legacy KNB/TetraTech solar-geometry PET disaggregation.
    Requires a ``lat`` coordinate on *template*.  Weights sum to 1.0 per
    coarse period (default: daily).

    Parameters
    ----------
    template : xr.DataArray
        DataArray providing the time range and ``lat`` coordinate.
        Typically a daily PET or similar variable.
    fine_freq : str
        Target fine temporal resolution for the output weights.
        Default ``"1h"`` (hourly).
    coarse_freq : str
        Coarse period over which weights must sum to 1.0.
        Default ``"1D"`` (daily).
    """
    fine_template = template.resample(time=fine_freq).ffill()
    lat = fine_template.coords["lat"]
    lat_rad = np.radians(lat)

    doy = fine_template.coords["time"].dt.dayofyear
    hour = fine_template.coords["time"].dt.hour

    # Solar declination and day length (legacy KNB empirical constants)
    declination = 0.40928 * np.cos(0.0172141 * (172.0 - doy))
    x2 = (-np.tan(lat_rad) * np.tan(declination)).clip(min=-1.0, max=1.0)
    day_length = 7.6394 * np.arccos(x2)
    sunrise = 12.0 - day_length / 2.0

    # Prevent division by zero for polar night
    valid_day = day_length > 0.0
    safe_day_length = day_length.where(valid_day, 1.0)

    dtr2 = safe_day_length / 2.0
    dtr4 = safe_day_length / 4.0
    crad = (2.0 / 3.0) / dtr2   # peak multiplier
    sl = crad / dtr4             # ramp slope

    tr2 = sunrise + dtr4
    tr3 = tr2 + dtr2
    tr4 = sunrise + safe_day_length  # sunset

    # Piecewise trapezoid
    fraction = xr.where(
        valid_day & (hour > sunrise) & (hour <= tr2),
        (hour - sunrise) * sl,
        0.0,
    )
    fraction = xr.where(
        valid_day & (hour > tr2) & (hour <= tr3),
        crad,
        fraction,
    )
    fraction = xr.where(
        valid_day & (hour > tr3) & (hour <= tr4),
        crad - (hour - tr3) * sl,
        fraction,
    )

    # Normalise so weights sum exactly to 1.0 per coarse period
    return _normalise_weights(fraction, coarse_freq).rename("trapezoidal_weights")


def _solar_weights(template: xr.DataArray, coarse_freq: str) -> xr.DataArray:
    """Generate normalised weights from pvlib clearsky radiation (Ineichen model)."""
    from met_timeseries.derivations import radiation

    clearsky = radiation.clearsky_radiation_ineichen(template)
    return _normalise_weights(clearsky.clip(min=0.0), coarse_freq)


# ---------------------------------------------------------------------------
# Weight resolver
# ---------------------------------------------------------------------------


def _resolve_weights(
    weight_method: WeightMethod,
    fine_pattern: xr.DataArray | None,
    weights: xr.DataArray | None,
    template: xr.DataArray | None,
    coarse_freq: str,
) -> xr.DataArray:
    """Resolve the appropriate weight array from the given method and inputs."""
    if weight_method == "proportional":
        if fine_pattern is None:
            raise ValueError("fine_pattern required for weight_method='proportional'")
        return _normalise_weights(fine_pattern, coarse_freq)
    elif weight_method == "trapezoidal":
        if template is None:
            raise ValueError(
                "template or fine_pattern required for weight_method='trapezoidal'"
            )
        return _trapezoidal_weights(template)
    elif weight_method == "solar":
        if template is None:
            raise ValueError(
                "template or fine_pattern required for weight_method='solar'"
            )
        return _solar_weights(template, coarse_freq)
    elif weight_method == "custom":
        if weights is None:
            raise ValueError("weights required for weight_method='custom'")
        return weights
    raise ValueError(f"Unknown weight_method: {weight_method!r}")


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------


def disaggregate(
    coarse: xr.DataArray,
    conservation: ConservationMethod,
    fine_pattern: xr.DataArray | None = None,
    weights: xr.DataArray | None = None,
    weight_method: WeightMethod = "proportional",
    template: xr.DataArray | None = None,
) -> xr.DataArray:
    """Disaggregate a coarse-resolution DataArray to finer resolution.

    Parameters
    ----------
    coarse : xr.DataArray
        Low-resolution DataArray.  Frequency inferred from ``time`` coordinate.
    conservation : ConservationMethod
        Physical conservation property of the variable:

        * ``"sum"``                  — extensive variables (precipitation, radiation)
          fine values sum back to coarse total each period.
        * ``"mean_additive"``        — intensive additive (temperature, dewpoint)
          fine mean matches coarse value, absolute differences preserved.
        * ``"mean_multiplicative"``  — intensive multiplicative (wind speed)
          fine mean matches coarse value, relative shape preserved.

    fine_pattern : xr.DataArray, optional
        High-resolution observed DataArray.  Required for ``mean_additive`` and
        ``mean_multiplicative``.  Required for ``conservation="sum"`` with
        ``weight_method="proportional"``.
    weights : xr.DataArray, optional
        Pre-computed normalised weights (sum to 1.0 per coarse period).
        Only used when ``weight_method="custom"``.
    weight_method : WeightMethod
        How to generate weights for sum-conserving disaggregation.
        Default ``"proportional"``.  Ignored for ``mean_additive`` /
        ``mean_multiplicative``.
    template : xr.DataArray, optional
        DataArray used only for time/lat coords when generating synthetic
        weights (``weight_method="trapezoidal"`` or ``"solar"``).  Falls back
        to *fine_pattern* if ``None``.
    """
    # Validate inputs before any computation so that error tests
    # don't fail with frequency-inference errors on trivially invalid inputs.
    if conservation not in ("sum", "mean_additive", "mean_multiplicative"):
        raise ValueError(f"Unknown conservation: {conservation!r}")

    if conservation == "mean_additive" and fine_pattern is None:
        raise ValueError("fine_pattern required for conservation='mean_additive'")

    if conservation == "mean_multiplicative" and fine_pattern is None:
        raise ValueError("fine_pattern required for conservation='mean_multiplicative'")

    if conservation == "sum" and weight_method == "proportional" and fine_pattern is None:
        raise ValueError("fine_pattern required for weight_method='proportional'")

    coarse_freq = _infer_freq(coarse)

    if conservation == "mean_additive":
        return _disaggregate_mean_additive(coarse, fine_pattern)

    if conservation == "mean_multiplicative":
        return _disaggregate_mean_multiplicative(coarse, fine_pattern)

    if conservation == "sum":
        resolved_weights = _resolve_weights(
            weight_method=weight_method,
            fine_pattern=fine_pattern,
            weights=weights,
            template=template if template is not None else fine_pattern,
            coarse_freq=coarse_freq,
        )
        return _disaggregate_sum(coarse, resolved_weights)

    raise ValueError(f"Unknown conservation: {conservation!r}")


# ---------------------------------------------------------------------------
# Named convenience wrappers — xarray API (thin, no logic)
# ---------------------------------------------------------------------------


def disaggregate_radiation(
    coarse: xr.DataArray, fine_pattern: xr.DataArray
) -> xr.DataArray:
    """Sum-conserving radiation disaggregation."""
    return disaggregate(coarse, conservation="sum", fine_pattern=fine_pattern)


def disaggregate_temperature(
    coarse: xr.DataArray, fine_pattern: xr.DataArray
) -> xr.DataArray:
    """Additive mean-preserving temperature disaggregation."""
    return disaggregate(coarse, conservation="mean_additive", fine_pattern=fine_pattern)


def disaggregate_dewpoint(
    coarse: xr.DataArray, fine_pattern: xr.DataArray
) -> xr.DataArray:
    """Additive mean-preserving dewpoint disaggregation.

    Additive (not multiplicative) because dewpoint depression is an absolute
    difference.
    """
    return disaggregate(coarse, conservation="mean_additive", fine_pattern=fine_pattern)


def disaggregate_wind(
    coarse: xr.DataArray, fine_pattern: xr.DataArray
) -> xr.DataArray:
    """Multiplicative mean-preserving wind disaggregation."""
    return disaggregate(
        coarse, conservation="mean_multiplicative", fine_pattern=fine_pattern
    )


def disaggregate_pet_trapezoidal(daily_pet: xr.DataArray) -> xr.DataArray:
    """Disaggregate daily PET using solar geometry trapezoid weights."""
    weights = _trapezoidal_weights(daily_pet)
    return disaggregate(
        daily_pet, conservation="sum", weight_method="custom", weights=weights
    ).rename("pet_disaggregated_trapezoidal")


def disaggregate_pet_solar(
    daily_pet: xr.DataArray, template: xr.DataArray
) -> xr.DataArray:
    """Disaggregate daily PET using pvlib clearsky radiation as weight template."""
    return disaggregate(
        daily_pet, conservation="sum", weight_method="solar", template=template
    ).rename("pet_disaggregated_solar")


# ---------------------------------------------------------------------------
# Stochastic precipitation wrapper (xarray interface over cascade internals)
# ---------------------------------------------------------------------------


def disaggregate_precipitation_stochastic(
    coarse: xr.DataArray,
    fine_pattern: xr.DataArray,
    method: Literal["molnar_burlando", "olsson"] = "molnar_burlando",
) -> xr.DataArray:
    """Stochastic precipitation disaggregation via cascade methods.

    Uses :func:`xr.apply_ufunc` to apply numpy/pandas cascade methods
    spatially.  *fine_pattern* is used as training data for cascade parameter
    fitting.

    Parameters
    ----------
    coarse : xr.DataArray
        Coarse (e.g. daily) precipitation DataArray.
    fine_pattern : xr.DataArray
        Fine (e.g. hourly) precipitation DataArray used for cascade fitting.
    method : {"molnar_burlando", "olsson"}
        ``"molnar_burlando"`` — log-Poisson canonical cascade
        (Molnar & Burlando, 2005).
        ``"olsson"`` — microcanonical empirical cascade (Olsson, 1998).
    """
    coarse_times = pd.DatetimeIndex(coarse.coords["time"].values)
    pattern_times = pd.DatetimeIndex(fine_pattern.coords["time"].values)
    # Use fine_pattern's own time axis as the output — the cascade methods
    # return a Series with the same index as pattern_times.

    def _apply_1d(coarse_vals: np.ndarray, pattern_vals: np.ndarray) -> np.ndarray:
        coarse_s = pd.Series(coarse_vals, index=coarse_times)
        pattern_s = pd.Series(pattern_vals, index=pattern_times)
        if method == "molnar_burlando":
            result = _molnar_burlando_disagg(coarse_s, pattern_s)
        elif method == "olsson":
            result = _olsson_disagg(coarse_s, pattern_s)
        else:
            raise ValueError(f"Unknown method: {method!r}")
        return result.values

    # Identify non-time dimensions for spatial iteration
    spatial_dims = [d for d in coarse.dims if d != "time"]

    if not spatial_dims:
        # 1-D case: apply directly
        result_vals = _apply_1d(coarse.values, fine_pattern.values)
        return xr.DataArray(
            result_vals,
            coords={"time": pattern_times},
            dims=["time"],
            attrs=coarse.attrs,
            name="precip_mm",
        )

    # nD case: iterate over spatial indices, applying cascade per grid cell
    result_shape = tuple(
        len(pattern_times) if d == "time" else coarse.sizes[d] for d in coarse.dims
    )
    result_vals = np.full(result_shape, np.nan)
    time_axis = list(coarse.dims).index("time")

    spatial_shape = tuple(coarse.sizes[d] for d in spatial_dims)
    for flat_idx in range(int(np.prod(spatial_shape))):
        spatial_idx = np.unravel_index(flat_idx, spatial_shape)
        # Build a full index tuple with time_axis as slice
        coarse_idx = list(spatial_idx)
        coarse_idx.insert(time_axis, slice(None))
        coarse_1d = coarse.values[tuple(coarse_idx)]
        pattern_1d = fine_pattern.values[tuple(coarse_idx)]
        hourly_1d = _apply_1d(coarse_1d, pattern_1d)
        result_idx = list(spatial_idx)
        result_idx.insert(time_axis, slice(None))
        result_vals[tuple(result_idx)] = hourly_1d

    out_coords = {
        d: (pattern_times if d == "time" else coarse.coords[d].values)
        for d in coarse.dims
    }
    return xr.DataArray(
        result_vals,
        coords=out_coords,
        dims=list(coarse.dims),
        attrs=coarse.attrs,
        name="precip_mm",
    )


# ---------------------------------------------------------------------------
# Hybrid precipitation — xarray vectorized version
# ---------------------------------------------------------------------------


def _disaggregate_precipitation_hybrid_xr(
    coarse: xr.DataArray,
    fine_pattern: xr.DataArray,
    cascade_method: str = "molnar_burlando",
) -> xr.DataArray:
    """Disaggregate daily precipitation using PRISM/NLDAS hybrid logic (xarray).

    Routing per day:
    - PRISM > 0 and NLDAS > 0 → proportional
    - PRISM == 0 and NLDAS > 0 → use NLDAS directly
    - PRISM > 0 and NLDAS == 0 → stochastic cascade
    - both zero → 0.0
    """
    coarse_freq = _infer_freq(coarse)

    # Precompute disaggregation results for all days
    proportional = disaggregate(coarse, conservation="sum", fine_pattern=fine_pattern)
    stochastic = disaggregate_precipitation_stochastic(
        coarse, fine_pattern, method=cascade_method
    )

    # Daily NLDAS sums broadcast to fine resolution
    nldas_daily = fine_pattern.resample(time=coarse_freq).sum()
    nldas_daily_h = nldas_daily.reindex_like(fine_pattern, method="ffill")
    coarse_h = coarse.reindex_like(fine_pattern, method="ffill")

    prism_gt0 = coarse_h > 0
    nldas_gt0 = nldas_daily_h > 0

    result = xr.where(
        prism_gt0 & nldas_gt0,
        proportional,
        xr.where(
            ~prism_gt0 & nldas_gt0,
            fine_pattern,
            xr.where(prism_gt0 & ~nldas_gt0, stochastic, 0.0),
        ),
    )
    return result.rename("precip_mm")


# ---------------------------------------------------------------------------
# Convenience: Hargreaves PET (pandas path, matches test expectations)
# ---------------------------------------------------------------------------


def pet_hargreaves(
    daily_tmin: pd.Series,
    daily_tmax: pd.Series,
    lat: float,
) -> pd.Series:
    """Estimate daily PET using the Hargreaves (1985) method.

    Thin wrapper around :func:`met_timeseries.derivations.pet.hargreaves`.

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
    from met_timeseries.derivations.pet import hargreaves

    return hargreaves(daily_tmin, daily_tmax, lat)
