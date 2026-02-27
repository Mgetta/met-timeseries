"""Tests for precipitation disaggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from met_timeseries.derivations.disaggregation import disaggregate_precip


def _make_prism_daily(daily_values, year=2020, month=1):
    """Create a minimal PRISM daily DataArray."""
    n_days = len(daily_values)
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=n_days, freq="D")
    return xr.DataArray(
        np.array(daily_values, dtype=float).reshape(n_days, 1, 1),
        coords={"time": dates, "lat": [35.0], "lon": [-84.0]},
        dims=["time", "lat", "lon"],
    )


def _make_nldas_hourly(hourly_values, year=2020, month=1):
    """Create a minimal NLDAS hourly DataArray."""
    n_hours = len(hourly_values)
    times = pd.date_range(f"{year}-{month:02d}-01", periods=n_hours, freq="h")
    return xr.DataArray(
        np.array(hourly_values, dtype=float).reshape(n_hours, 1, 1),
        coords={"time": times, "y": [35.0], "x": [-84.0]},
        dims=["time", "y", "x"],
    )


def test_disaggregation_preserves_daily_total():
    """Daily sum of hourly output should equal PRISM daily total."""
    daily_total = 10.0  # mm/day
    prism = _make_prism_daily([daily_total])
    # NLDAS pattern: 8mm in first hour, rest zero (concentrating all in one hour)
    nldas_vals = [8.0] + [0.0] * 23
    nldas = _make_nldas_hourly(nldas_vals)
    result = disaggregate_precip(prism, nldas)
    # Daily sum of result should equal PRISM daily total
    daily_sum = float(result.sum("time").values[0, 0])
    np.testing.assert_allclose(daily_sum, daily_total, rtol=1e-5)


def test_disaggregation_uniform_when_nldas_zero():
    """When NLDAS hourly precip is all zero, use uniform distribution."""
    prism = _make_prism_daily([24.0])
    nldas = _make_nldas_hourly([0.0] * 24)
    result = disaggregate_precip(prism, nldas)
    # All hours should be equal
    vals = result.values[:, 0, 0]
    np.testing.assert_allclose(vals, np.full(24, 1.0), rtol=1e-5)


def test_disaggregation_output_shape():
    prism = _make_prism_daily([5.0, 3.0])  # 2 days
    nldas = _make_nldas_hourly([1.0] * 48)  # 48 hours
    result = disaggregate_precip(prism, nldas)
    assert result.dims == ("time", "lat", "lon")
    assert result.sizes["time"] == 48
