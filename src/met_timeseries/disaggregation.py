"""
Source-agnostic temporal disaggregation functions.

Each function is a pure transformation on :class:`pandas.Series` with a
:class:`pandas.DatetimeIndex` — no knowledge of NLDAS/PRISM variable names or
dataset structure.  The caller pairs the correct daily and hourly series.

General pattern: **daily truth × hourly shape → hourly output**.

The caller is responsible for ensuring the daily and hourly series are
temporally aligned (same days, consistent index frequency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import poisson
from mettoolbox.melodist.melodist.precipitation import build_casc, disagg_prec
from scipy.optimize import minimize

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
    daily_total: pd.Series,
    hourly_pattern: pd.Series,
    method: str = "molnar_burlando",
) -> pd.Series:
    """Disaggregate daily precipitation to hourly using a cascade method.

    Parameters
    ----------
    daily_total:
        Daily precipitation totals in mm with a daily
        :class:`~pandas.DatetimeIndex`.
    hourly_pattern:
        Hourly precipitation observations (e.g. from NLDAS-2) used
        for cascade parameter fitting.
    method:
        ``"molnar_burlando"`` — canonical log-Poisson cascade
        (Molnar & Burlando, 2005).
        ``"olsson"`` — microcanonical empirical cascade
        (Olsson, 1998) via MELODIST.

    Returns
    -------
    :class:`pandas.Series` named ``precip_mm``.
    """
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



