"""
Source-agnostic temporal disaggregation.

Core API
--------
disaggregate(coarse, conservation, ...)
    Generic disaggregation. Conservation property determines the math.
    Weights are resolved dynamically based on the weight_method.

Convenience wrappers
--------------------
disaggregate_precipitation, disaggregate_radiation, disaggregate_temperature,
disaggregate_dewpoint, disaggregate_wind, disaggregate_pet_trapezoidal,
disaggregate_pet_solar, disaggregate_pevt

All operate on xr.DataArray with a 'time' coordinate.
Frequencies are inferred automatically — no hardcoded "1h" or "1D".
"""

from __future__ import annotations

from functools import partial
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import poisson
from mettoolbox.melodist.melodist.precipitation import build_casc, disagg_prec

# ===========================================================================
# PRIMARY PUBLIC APIS
# ===========================================================================

def disaggregate(
    coarse: xr.DataArray,
    conservation: ConservationMethod,
    fine_pattern: xr.DataArray | None = None,
    weights: xr.DataArray | None = None,
    weight_method: WeightMethod = "proportional",
    template: xr.DataArray | None = None,
    offset: pd.Timedelta | None = None
) -> xr.DataArray:
    """Disaggregate a coarse-resolution DataArray to finer resolution."""
    disagg_func = _CONSERVATION_DISPATCH.get(conservation)
    if not disagg_func:
        raise ValueError(f"Unknown conservation {conservation!r}. Options: {list(_CONSERVATION_DISPATCH.keys())}")

    return disagg_func(
        coarse=coarse, 
        fine_pattern=fine_pattern, 
        weights=weights, 
        weight_method=weight_method, 
        template=template, 
        coarse_freq=_infer_freq(coarse),
        offset = offset
    )

def disaggregate_pevt(
    pevt_daily: xr.DataArray, 
    method: Literal["diurnal", "static"] = "diurnal",
    custom_weights: list[float] | np.ndarray | None = None
) -> xr.DataArray:
    """Downscale daily PEVT into hourly values."""
    weight_func = _PEVT_DISPATCH.get(method)
    if not weight_func:
        raise ValueError(f"Unknown PEVT method {method!r}. Options: {list(_PEVT_DISPATCH.keys())}")

    weights = weight_func(template=pevt_daily, custom_weights=custom_weights)
    pevt_hourly = disaggregate(pevt_daily, conservation="sum", weight_method="custom", weights=weights)
    
    pevt_hourly.attrs["units"] = pevt_daily.attrs.get("units", "mm/hour")
    pevt_hourly.attrs["description"] = f"PEVT downscaled using '{method}' method."
    return pevt_hourly.rename("pevt_hourly")

def disaggregate_precipitation(
    coarse: xr.DataArray,
    fine_pattern: xr.DataArray,
    method: str = "molnar_burlando",
) -> xr.DataArray:
    """Disaggregate daily precipitation to finer resolution."""
    disagg_func = _PRECIP_DISPATCH.get(method)
    if not disagg_func:
        raise ValueError(f"Unknown precipitation method {method!r}. Options: {list(_PRECIP_DISPATCH.keys())}")

    return disagg_func(coarse=coarse, fine_pattern=fine_pattern)

# ---------------------------------------------------------------------------
# Named convenience wrappers 
# ---------------------------------------------------------------------------
disaggregate_radiation = partial(disaggregate, conservation="sum")
disaggregate_temperature = partial(disaggregate, conservation="mean_additive")
disaggregate_dewpoint = partial(disaggregate, conservation="mean_additive")
disaggregate_wind = partial(disaggregate, conservation="mean_multiplicative")
disaggregate_pet_trapezoidal = partial(disaggregate, conservation="sum", weight_method="trapezoidal")
disaggregate_pet_solar = partial(disaggregate, conservation="sum", weight_method="solar")



# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ConservationMethod = Literal["sum", "mean_additive", "mean_multiplicative"]
WeightMethod = Literal["proportional", "trapezoidal", "solar", "custom"]

# ---------------------------------------------------------------------------
# Cascade internal helpers (1D math wrapped by xarray apply_ufunc)
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
    # Initialize a local, unpredictable RNG for this specific chunk
    rng = np.random.default_rng() 
    
    n = len(values)
    out = np.zeros(n * 2)

    prob_dry = 1.0 - 2 ** (-1.0 + tau_obs[0])
    prob_dry = prob_dry * (1 + prob_dry)

    for i in range(n):
        if values[i] == 0:
            continue

        # Use the local rng instead of scipy.stats.poisson or np.random
        W = A * (beta ** rng.poisson(1, size=2))
        W[W < 0] = 0

        wet = rng.binomial(1, 1 - prob_dry, size=2)
        # ... rest of the logic ...
        if wet[0] == 0 and wet[1] == 0:
            wet[rng.integers(2)] = 1

        W = W * wet
        w_sum = W[0] + W[1]
        if w_sum <= 0:
            W[:] = 0.5
            w_sum = 1.0

        out[2 * i] = values[i] * W[0] / w_sum
        out[2 * i + 1] = values[i] * W[1] / w_sum

    return out

def _cascade_to_hourly(values: np.ndarray, daily_total: pd.Series) -> pd.Series:
    """Convert 45-min cascade output to hourly via Güntner et al. (2001)."""
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

# ===========================================================================
# xarray-native, frequency-agnostic disaggregation framework
# ===========================================================================

# ---------------------------------------------------------------------------
# Frequency inference helpers
# ---------------------------------------------------------------------------


def _infer_freq(da: xr.DataArray, fallback_single_step: str = "1D") -> str:
    """Infer time frequency string from a DataArray's time coordinate."""
    times = pd.DatetimeIndex(da.coords["time"].values)
    
    # --- The Fix: If there is only one timestamp, assume the fallback (Daily) ---
    if len(times) == 1:
        return fallback_single_step
        
    freq = pd.infer_freq(times)
    
    if freq is None:
        name_hint = f" '{da.name}'" if da.name else ""
        time_hint = f" (first timestamp: {times[0]})" if len(times) else ""
        raise ValueError(
            f"Cannot infer time frequency for DataArray{name_hint}{time_hint}"
            " — check for gaps or irregular spacing."
        )
    return freq

def _count_fine_per_coarse(fine: xr.DataArray, coarse_freq: str) -> xr.DataArray:
    """Count fine timesteps per coarse period, broadcast back to fine resolution."""
    counts = xr.ones_like(fine).resample(time=coarse_freq).sum()
    return counts.reindex_like(fine, method="ffill")

# ---------------------------------------------------------------------------
# Private core implementations (xarray-native, vectorized)
# ---------------------------------------------------------------------------
def _resample_with_offset(da: xr.DataArray, freq: str, offset: pd.Timedelta, how: str = "sum") -> xr.DataArray:
    """Safely resample a DataArray with a time offset, immune to xarray/pandas version changes."""
    # 1. Shift time backward so bins align natively to midnight
    shifted_time = da.coords["time"] - offset
    da_shifted = da.assign_coords(time=shifted_time)
    
    # 2. Resample at midnight (guaranteed safe left-labeling)
    resampled = getattr(da_shifted.resample(time=freq), how)()
    
    # 3. Shift labels back forward to match the original PRISM offset
    restored_time = resampled.coords["time"] + offset
    return resampled.assign_coords(time=restored_time)

def _disaggregate_sum(coarse: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """Sum-conserving disaggregation: fine values sum back to the coarse total."""
    coarse_h = coarse.reindex_like(weights, method="ffill")
    return (coarse_h * weights).clip(min=0.0)

def _disaggregate_mean_additive(coarse: xr.DataArray, fine_pattern: xr.DataArray, **kwargs) -> xr.DataArray:
    """Additive bias correction: shift fine pattern so its mean matches coarse value."""
    if fine_pattern is None: raise ValueError("fine_pattern required for conservation='mean_additive'")
    coarse_freq = _infer_freq(coarse)
    pattern_mean = fine_pattern.resample(time=coarse_freq).mean()
    bias = coarse - pattern_mean
    bias_h = bias.reindex_like(fine_pattern, method="ffill")
    return fine_pattern + bias_h

def _disaggregate_mean_multiplicative(coarse: xr.DataArray, fine_pattern: xr.DataArray, **kwargs) -> xr.DataArray:
    """Multiplicative scaling: scale fine pattern so its mean matches coarse value."""
    if fine_pattern is None: raise ValueError("fine_pattern required for conservation='mean_multiplicative'")
    coarse_freq = _infer_freq(coarse)
    pattern_mean = fine_pattern.resample(time=coarse_freq).mean()
    scale = xr.where(pattern_mean > 0, coarse / pattern_mean, np.nan)
    scale_h = scale.reindex_like(fine_pattern, method="ffill")
    coarse_h = coarse.reindex_like(fine_pattern, method="ffill")
    return xr.where(scale_h.isnull(), coarse_h, (fine_pattern * scale_h).clip(min=0.0))

# ---------------------------------------------------------------------------
# Weight generators & PEVT Logic
# ---------------------------------------------------------------------------

def _weights_normalise_respec(fine_pattern: xr.DataArray, coarse_freq: str) -> xr.DataArray:
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

def _weights_normalise(fine_pattern: xr.DataArray, coarse_freq: str, offset: pd.Timedelta) -> xr.DataArray:
    """Normalise a fine-resolution array so it sums to 1.0 per coarse period."""
    # Use the robust resampler instead of resample(offset=...)
    period_sum = _resample_with_offset(fine_pattern, coarse_freq, offset, "sum")
    safe_sum = period_sum.where(period_sum > 0, other=1.0)
    
    safe_sum_h = safe_sum.reindex_like(fine_pattern, method="ffill")
    period_sum_h = period_sum.reindex_like(fine_pattern, method="ffill")
    
    counts = xr.ones_like(fine_pattern)
    n_fine = _resample_with_offset(counts, coarse_freq, offset, "sum").reindex_like(fine_pattern, method="ffill")
    
    return xr.where(period_sum_h > 0, fine_pattern / safe_sum_h, 1.0 / n_fine)

def _weights_trapezoidal(template: xr.DataArray, coarse_freq: str, offset: pd.Timedelta, fine_freq: str = "1h") -> xr.DataArray:
    """Generate normalised weights from solar trapezoid geometry."""
    fine_template = template.resample(time=fine_freq).ffill()
    lat_rad = np.radians(fine_template.coords["lat"])
    doy = fine_template.coords["time"].dt.dayofyear
    hour = fine_template.coords["time"].dt.hour

    declination = 0.40928 * np.cos(0.0172141 * (172.0 - doy))
    x2 = (-np.tan(lat_rad) * np.tan(declination)).clip(min=-1.0, max=1.0)
    day_length = 7.6394 * np.arccos(x2)
    sunrise = 12.0 - day_length / 2.0

    valid_day = day_length > 0.0
    safe_day_length = day_length.where(valid_day, 1.0)

    dtr2, dtr4 = safe_day_length / 2.0, safe_day_length / 4.0
    crad = (2.0 / 3.0) / dtr2
    sl = crad / dtr4

    tr2, tr3, tr4 = sunrise + dtr4, sunrise + dtr4 + dtr2, sunrise + safe_day_length

    fraction = xr.where(valid_day & (hour > sunrise) & (hour <= tr2), (hour - sunrise) * sl, 0.0)
    fraction = xr.where(valid_day & (hour > tr2) & (hour <= tr3), crad, fraction)
    fraction = xr.where(valid_day & (hour > tr3) & (hour <= tr4), crad - (hour - tr3) * sl, fraction)

    return _weights_normalise(fraction, coarse_freq, offset).rename("trapezoidal_weights")

def _weights_solar(template: xr.DataArray, coarse_freq: str, offset: pd.Timedelta) -> xr.DataArray:
    """Generate normalised weights from pvlib clearsky radiation."""
    from met_timeseries.derivations import radiation
    clearsky = radiation.clearsky_radiation_ineichen(template)
    return _weights_normalise(clearsky.clip(min=0.0), coarse_freq, offset)

def _weights_diurnal(
    template: xr.DataArray, 
    coarse_freq: str = "1D", 
    offset: pd.Timedelta = pd.Timedelta(0), 
    custom_weights: list[float] | np.ndarray | None = None, 
    **kwargs
) -> xr.DataArray:
    """Generate normalised diurnal weights."""
    if custom_weights is None:
        custom_weights = [
            0.05, 0.05, 0.03, 0.03, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.15, 0.10,
            0.08, 0.07, 0.05, 0.03, 0.02, 0.03, 0.03, 0.05, 0.07, 0.08, 0.05, 0.05
        ]
    weights_arr = np.asarray(custom_weights, dtype=float)
    if len(weights_arr) != 24:
        raise ValueError("Diurnal weights must contain exactly 24 elements.")

    fine_template = template.resample(time="1H").ffill()
    fraction = xr.ones_like(fine_template) * weights_arr[fine_template.coords["time"].dt.hour.values]
    
    return _weights_normalise(fraction, coarse_freq, offset).rename("diurnal_weights")


def _weights_static(
    template: xr.DataArray, 
    coarse_freq: str = "1D", 
    offset: pd.Timedelta = pd.Timedelta(0), 
    **kwargs
) -> xr.DataArray:
    """Generate flat normalized weights."""
    flat_template = template.resample(time="1H").ffill()
    
    return _weights_normalise(xr.ones_like(flat_template), coarse_freq, offset).rename("static_weights")
# ---------------------------------------------------------------------------
# Precipitation Logic
# ---------------------------------------------------------------------------

def disaggregate_precipitation_stochastic(
    coarse: xr.DataArray,
    fine_pattern: xr.DataArray,
    method: Literal["molnar_burlando", "olsson"] = "molnar_burlando",
) -> xr.DataArray:
    """Stochastic precipitation disaggregation via cascade methods."""
    coarse_times = pd.DatetimeIndex(coarse.coords["time"].values)
    pattern_times = pd.DatetimeIndex(fine_pattern.coords["time"].values)

    def _apply_1d(coarse_vals: np.ndarray, pattern_vals: np.ndarray) -> np.ndarray:
        if np.isnan(coarse_vals).all():
            return np.full(len(pattern_vals), np.nan)
        
        coarse_safe = np.nan_to_num(coarse_vals, nan=0.0)
        coarse_s = pd.Series(coarse_safe, index=coarse_times)
        pattern_s = pd.Series(pattern_vals, index=pattern_times)

        if method == "molnar_burlando":
            result = _molnar_burlando_disagg(coarse_s, pattern_s)
        elif method == "olsson":
            result = _olsson_disagg(coarse_s, pattern_s)
            
        # If the result is a DataFrame (e.g. from olsson), extract the series
        if isinstance(result, pd.DataFrame):
            result = result.iloc[:, 0]
            
        # Reindex to exactly match the fine_pattern timestamps. 
        # Missing edges caused by timezone shifts will be filled with 0.0 (no rain).
        result = result.reindex(pattern_times).fillna(0.0)
            
        return result.values

    result = xr.apply_ufunc(
        _apply_1d,
        coarse.rename({"time": "time_coarse"}),
        fine_pattern,
        input_core_dims=[["time_coarse"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"time": len(pattern_times)},
                            "allow_rechunk": True}
    )

    result = result.assign_coords({"time": pattern_times})
    result.attrs = coarse.attrs
    return result.rename("precip_mm")

def disaggregate_precipitation_hybrid(
    coarse: xr.DataArray,
    fine_pattern: xr.DataArray,
    cascade_method: str = "molnar_burlando",
    enforce_coarse_mass: bool = True
) -> xr.DataArray:
    """Disaggregate daily precipitation using PRISM/NLDAS hybrid logic."""
    coarse_freq = _infer_freq(coarse)
    coarse_hour = pd.Timestamp(coarse.coords["time"].values[0]).hour
    offset = pd.Timedelta(hours=coarse_hour)

    # 1. Use robust resampler to guarantee perfect time alignment
    nldas_daily_h = _resample_with_offset(fine_pattern, coarse_freq, offset, "sum").reindex_like(fine_pattern, method="ffill")
    coarse_h = coarse.reindex_like(fine_pattern, method="ffill")

    # 2. Evaluate logic with strict NaN protection
    prism_valid = coarse_h.notnull()
    prism_gt0 = (coarse_h > 0) & prism_valid
    prism_zero = (coarse_h == 0) & prism_valid  
    nldas_gt0 = nldas_daily_h > 0
    
    mask_prop_h = prism_gt0 & nldas_gt0
    mask_stoch_h = prism_gt0 & ~nldas_gt0
    mask_direct_h = prism_zero & nldas_gt0

    result = xr.zeros_like(fine_pattern, dtype=float)

    # --- THE FIX: Remove the .any() if statements ---
    
    # 1. Proportional Math Blueprint
    proportional = disaggregate(coarse, conservation="sum", fine_pattern=fine_pattern)
    result = xr.where(mask_prop_h, proportional, result)

    # 2. Stochastic Math Blueprint
    mask_stoch_daily = mask_stoch_h.reindex_like(coarse, method="nearest").fillna(False)
    coarse_stoch = xr.where(mask_stoch_daily, coarse, 0.0)
    stochastic = disaggregate_precipitation_stochastic(coarse_stoch, fine_pattern, method=cascade_method)
    result = xr.where(mask_stoch_h, stochastic, result)

    # 3. Direct Math Blueprint
    if not enforce_coarse_mass:
        # We can keep the boolean check on the static python variable 'enforce_coarse_mass'
        # because it does not depend on the Dask array.
        result = xr.where(mask_direct_h, fine_pattern, result)

    return result.fillna(0.0).rename("precip_mm")

def _disaggregate_precipitation_hybrid(
    coarse: xr.DataArray,
    fine_pattern: xr.DataArray,
    cascade_method: str = "molnar_burlando",
    enforce_coarse_mass: bool = True
) -> xr.DataArray:
    """Disaggregate daily precipitation using PRISM/NLDAS hybrid logic."""
    coarse_freq = _infer_freq(coarse)
    coarse_hour = pd.Timestamp(coarse.coords["time"].values[0]).hour
    offset = pd.Timedelta(hours=coarse_hour)

    # 1. Use robust resampler to guarantee perfect time alignment
    nldas_daily_h = _resample_with_offset(fine_pattern, coarse_freq, offset, "sum").reindex_like(fine_pattern, method="ffill")
    coarse_h = coarse.reindex_like(fine_pattern, method="ffill")

    # 2. Evaluate logic with strict NaN protection
    prism_valid = coarse_h.notnull()
    prism_gt0 = (coarse_h > 0) & prism_valid
    prism_zero = (coarse_h == 0) & prism_valid  # Strictly 0.0, not NaN
    nldas_gt0 = nldas_daily_h > 0
    
    mask_prop_h = prism_gt0 & nldas_gt0
    mask_stoch_h = prism_gt0 & ~nldas_gt0
    mask_direct_h = prism_zero & nldas_gt0

    result = xr.zeros_like(fine_pattern, dtype=float)

    if mask_prop_h.any():
        proportional = disaggregate(coarse, conservation="sum", fine_pattern=fine_pattern)
        result = xr.where(mask_prop_h, proportional, result)

    if mask_stoch_h.any():
        # Optimization to avoid stochastic math on every single day: first identify which days need it, then only apply the expensive cascade to those days.
        # Downsample the hourly mask back to the exact daily coordinates of coarse.
        # Because mask_stoch_h is constant for the whole day, taking the "nearest" hour safely grabs that day's boolean state.
        mask_stoch_daily = mask_stoch_h.reindex_like(coarse, method="nearest").fillna(False)
        
        # Zero out days that don't need stochastic math to trigger the shortcut in _molnar_burlando_split
        coarse_stoch = xr.where(mask_stoch_daily, coarse, 0.0)
        
        stochastic = disaggregate_precipitation_stochastic(coarse_stoch, fine_pattern, method=cascade_method)
        result = xr.where(mask_stoch_h, stochastic, result)

    if mask_direct_h.any() and not enforce_coarse_mass:
        result = xr.where(mask_direct_h, fine_pattern, result)

    return result.fillna(0.0).rename("precip_mm")

# ===========================================================================
# DICTIONARY REGISTRIES
# ===========================================================================

# 1. Weights Registry
_WEIGHT_DISPATCH = {
    "proportional": lambda fp, c_freq, offset, **kw: _weights_normalise(fp, c_freq, offset) if fp is not None else (_ for _ in ()).throw(ValueError("fine_pattern required")),
    "trapezoidal": lambda tmpl, fp, c_freq, offset, **kw: _weights_trapezoidal(tmpl if tmpl is not None else fp, c_freq, offset) if (tmpl is not None or fp is not None) else (_ for _ in ()).throw(ValueError("template/fine_pattern required")),
    "solar": lambda tmpl, fp, c_freq, offset, **kw: _weights_solar(tmpl if tmpl is not None else fp, c_freq, offset) if (tmpl is not None or fp is not None) else (_ for _ in ()).throw(ValueError("template/fine_pattern required")),
    "custom": lambda weights, **kw: weights if weights is not None else (_ for _ in ()).throw(ValueError("weights required for 'custom'")),
}

# 2. Conservation Registry
def _run_sum_conservation(coarse, fine_pattern, weights, weight_method, template, coarse_freq, offset = None):
    # Extract the timezone/timestamp offset
    coarse_hour = pd.Timestamp(coarse.coords["time"].values[0]).hour
    if offset is None:
        offset = pd.Timedelta(hours=coarse_hour)

    weight_func = _WEIGHT_DISPATCH.get(weight_method)
    if not weight_func:
        raise ValueError(f"Unknown weight_method {weight_method!r}.")
        
    resolved_weights = weight_func(fp=fine_pattern, tmpl=template, c_freq=coarse_freq, weights=weights, offset=offset)
    return _disaggregate_sum(coarse, resolved_weights)

_CONSERVATION_DISPATCH = {
    "sum": _run_sum_conservation,
    "mean_additive": _disaggregate_mean_additive,
    "mean_multiplicative": _disaggregate_mean_multiplicative,
}

# 3. PEVT & Precipitation Registries
_PEVT_DISPATCH = {
    "static": _weights_static,
    "diurnal": _weights_diurnal,
}

_PRECIP_DISPATCH = {
    "proportional": partial(disaggregate, conservation="sum"),
    "molnar_burlando": partial(disaggregate_precipitation_stochastic, method="molnar_burlando"),
    "olsson": partial(disaggregate_precipitation_stochastic, method="olsson"),
    "hybrid": partial(disaggregate_precipitation_hybrid, cascade_method="molnar_burlando"),
}

