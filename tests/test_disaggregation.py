"""
Tests for met_timeseries.disaggregation module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_series(values, start="2000-01-01"):
    """Create a daily pd.Series from a list of values."""
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def _hourly_series(values, start="2000-01-01"):
    """Create an hourly pd.Series from a list of values."""
    index = pd.date_range(start, periods=len(values), freq="h")
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# _proportional_disaggregate
# ---------------------------------------------------------------------------


class TestProportionalDisaggregate:
    def test_basic_proportional(self):
        from met_timeseries.disaggregation import _proportional_disaggregate

        # 1 day, 4 hours, equal pattern → each hour gets 0.25 * daily_total
        daily = _daily_series([8.0])
        pattern = _hourly_series([1.0, 1.0, 1.0, 1.0])
        result = _proportional_disaggregate(daily, pattern, name="test")
        np.testing.assert_allclose(result.values, [2.0, 2.0, 2.0, 2.0])

    def test_zero_pattern_spreads_equally(self):
        from met_timeseries.disaggregation import _proportional_disaggregate

        daily = _daily_series([12.0])
        pattern = _hourly_series([0.0, 0.0, 0.0])
        result = _proportional_disaggregate(daily, pattern, name="test")
        np.testing.assert_allclose(result.values, [4.0, 4.0, 4.0])

    def test_name_assigned(self):
        from met_timeseries.disaggregation import _proportional_disaggregate

        daily = _daily_series([1.0])
        pattern = _hourly_series([1.0, 1.0])
        result = _proportional_disaggregate(daily, pattern, name="my_var")
        assert result.name == "my_var"

    def test_multi_day(self):
        from met_timeseries.disaggregation import _proportional_disaggregate

        # 2 days × 2 hours each
        daily = pd.Series(
            [4.0, 6.0],
            index=pd.date_range("2000-01-01", periods=2, freq="D"),
        )
        pattern = pd.Series(
            [1.0, 3.0, 2.0, 2.0],
            index=pd.date_range("2000-01-01", periods=4, freq="12h"),
        )
        result = _proportional_disaggregate(daily, pattern, name="p")
        assert abs(float(result.iloc[0]) - 1.0) < 1e-10  # 4 * (1/4)
        assert abs(float(result.iloc[1]) - 3.0) < 1e-10  # 4 * (3/4)
        assert abs(float(result.iloc[2]) - 3.0) < 1e-10  # 6 * (2/4)
        assert abs(float(result.iloc[3]) - 3.0) < 1e-10  # 6 * (2/4)


# ---------------------------------------------------------------------------
# disaggregate_precipitation
# ---------------------------------------------------------------------------


class TestDisaggregatePrecipitation:
    def test_name(self):
        from met_timeseries.disaggregation import disaggregate_precipitation

        daily = _daily_series([5.0])
        pattern = _hourly_series([1.0, 2.0])
        result = disaggregate_precipitation(daily, pattern)
        assert result.name == "precip_mm"

    def test_sums_to_daily_total(self):
        from met_timeseries.disaggregation import disaggregate_precipitation

        daily = _daily_series([10.0])
        pattern = _hourly_series([1.0, 3.0, 2.0, 4.0])
        result = disaggregate_precipitation(daily, pattern)
        assert abs(result.sum() - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# disaggregate_temperature_pattern
# ---------------------------------------------------------------------------


class TestDisaggregateTemperaturePattern:
    def test_rescales_to_tmin_tmax(self):
        from met_timeseries.disaggregation import disaggregate_temperature_pattern

        daily_tmin = _daily_series([10.0])
        daily_tmax = _daily_series([20.0])
        # Pattern has 4 equally-spaced values; min=0, max=3
        pattern = _hourly_series([0.0, 1.0, 2.0, 3.0])
        result = disaggregate_temperature_pattern(daily_tmin, daily_tmax, pattern)
        assert result.name == "temp_c"
        # Min should map to tmin, max should map to tmax
        assert abs(float(result.iloc[0]) - 10.0) < 1e-10
        assert abs(float(result.iloc[-1]) - 20.0) < 1e-10

    def test_flat_pattern_uses_mean(self):
        from met_timeseries.disaggregation import disaggregate_temperature_pattern

        daily_tmin = _daily_series([8.0])
        daily_tmax = _daily_series([18.0])
        pattern = _hourly_series([5.0, 5.0, 5.0])  # flat
        result = disaggregate_temperature_pattern(daily_tmin, daily_tmax, pattern)
        np.testing.assert_allclose(result.values, 13.0)  # (8+18)/2

    def test_name(self):
        from met_timeseries.disaggregation import disaggregate_temperature_pattern

        result = disaggregate_temperature_pattern(
            _daily_series([5.0]), _daily_series([15.0]), _hourly_series([1.0, 2.0])
        )
        assert result.name == "temp_c"


# ---------------------------------------------------------------------------
# disaggregate_radiation_pattern
# ---------------------------------------------------------------------------


class TestDisaggregateRadiationPattern:
    def test_name(self):
        from met_timeseries.disaggregation import disaggregate_radiation_pattern

        result = disaggregate_radiation_pattern(_daily_series([100.0]), _hourly_series([1.0, 1.0]))
        assert result.name == "shortwave_wm2"

    def test_sums_to_daily(self):
        from met_timeseries.disaggregation import disaggregate_radiation_pattern

        daily = _daily_series([200.0])
        pattern = _hourly_series([1.0, 2.0, 3.0, 4.0])
        result = disaggregate_radiation_pattern(daily, pattern)
        assert abs(result.sum() - 200.0) < 1e-10


# ---------------------------------------------------------------------------
# disaggregate_wind_pattern
# ---------------------------------------------------------------------------


class TestDisaggregateWindPattern:
    def test_scales_to_daily_mean(self):
        from met_timeseries.disaggregation import disaggregate_wind_pattern

        daily = _daily_series([6.0])
        pattern = _hourly_series([2.0, 4.0, 6.0, 8.0])  # mean = 5
        result = disaggregate_wind_pattern(daily, pattern)
        assert result.name == "wind_speed_ms"
        assert abs(float(result.mean()) - 6.0) < 1e-10

    def test_clips_to_zero(self):
        from met_timeseries.disaggregation import disaggregate_wind_pattern

        daily = _daily_series([0.0])
        pattern = _hourly_series([1.0, 2.0])
        result = disaggregate_wind_pattern(daily, pattern)
        assert np.all(result.values >= 0.0)

    def test_name(self):
        from met_timeseries.disaggregation import disaggregate_wind_pattern

        result = disaggregate_wind_pattern(_daily_series([3.0]), _hourly_series([1.0, 1.0]))
        assert result.name == "wind_speed_ms"


# ---------------------------------------------------------------------------
# disaggregate_wind_equal
# ---------------------------------------------------------------------------


class TestDisaggregateWindEqual:
    def test_constant_within_day(self):
        from met_timeseries.disaggregation import disaggregate_wind_equal

        daily = _daily_series([5.0])
        result = disaggregate_wind_equal(daily)
        assert result.name == "wind_speed_ms"
        assert np.all(result.values == 5.0)

    def test_hourly_length(self):
        from met_timeseries.disaggregation import disaggregate_wind_equal

        daily = _daily_series([3.0, 4.0])
        result = disaggregate_wind_equal(daily)
        assert len(result) == 48  # 2 days × 24 hours


# ---------------------------------------------------------------------------
# disaggregate_dewpoint_pattern
# ---------------------------------------------------------------------------


class TestDisaggregateDewpointPattern:
    def test_bias_correction(self):
        from met_timeseries.disaggregation import disaggregate_dewpoint_pattern

        daily = _daily_series([10.0])
        pattern = _hourly_series([8.0, 9.0, 10.0, 11.0])  # mean = 9.5
        result = disaggregate_dewpoint_pattern(daily, pattern)
        # bias = 10.0 - 9.5 = 0.5; each hour + 0.5
        np.testing.assert_allclose(result.values, [8.5, 9.5, 10.5, 11.5])

    def test_name(self):
        from met_timeseries.disaggregation import disaggregate_dewpoint_pattern

        result = disaggregate_dewpoint_pattern(_daily_series([5.0]), _hourly_series([5.0, 5.0]))
        assert result.name == "dewpoint_c"


# ---------------------------------------------------------------------------
# disaggregate_dewpoint_constant
# ---------------------------------------------------------------------------


class TestDisaggregateDewpointConstant:
    def test_constant_within_day(self):
        from met_timeseries.disaggregation import disaggregate_dewpoint_constant

        daily = _daily_series([12.0])
        result = disaggregate_dewpoint_constant(daily)
        assert result.name == "dewpoint_c"
        assert np.all(result.values == 12.0)

    def test_hourly_length(self):
        from met_timeseries.disaggregation import disaggregate_dewpoint_constant

        daily = _daily_series([5.0, 6.0])
        result = disaggregate_dewpoint_constant(daily)
        assert len(result) == 48


# ---------------------------------------------------------------------------
# pet_hargreaves
# ---------------------------------------------------------------------------


class TestPetHargreaves:
    def test_positive_pet(self):
        from met_timeseries.disaggregation import pet_hargreaves

        daily_tmin = _daily_series([10.0])
        daily_tmax = _daily_series([25.0])
        result = pet_hargreaves(daily_tmin, daily_tmax, lat=45.0)
        assert result.name == "pet_hargreaves_mm"
        assert float(result.iloc[0]) > 0.0

    def test_no_negative_pet(self):
        from met_timeseries.disaggregation import pet_hargreaves

        daily_tmin = _daily_series([-30.0])
        daily_tmax = _daily_series([-25.0])
        result = pet_hargreaves(daily_tmin, daily_tmax, lat=70.0)
        assert np.all(result.values >= 0.0)

    def test_name(self):
        from met_timeseries.disaggregation import pet_hargreaves

        result = pet_hargreaves(_daily_series([5.0]), _daily_series([20.0]), lat=40.0)
        assert result.name == "pet_hargreaves_mm"
