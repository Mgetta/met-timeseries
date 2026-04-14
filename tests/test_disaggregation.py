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

    def test_day_start_hour_shifts_day_boundary(self):
        """With day_start_hour=12, hours 12-23 of day-0 belong to day-1."""
        from met_timeseries.disaggregation import disaggregate_precipitation

        # 2 daily values starting Jan 1
        daily = pd.Series(
            [10.0, 20.0],
            index=pd.date_range("2000-01-01", periods=2, freq="D"),
        )
        # 48 hourly values: hours 0..23 of Jan 1, hours 0..23 of Jan 2
        pattern = pd.Series(
            [1.0] * 48,
            index=pd.date_range("2000-01-01", periods=48, freq="h"),
        )
        result = disaggregate_precipitation(daily, pattern, day_start_hour=12)
        # With day_start_hour=12:
        #   Jan-1 day label covers hours 12:00 Jan-1 through 11:00 Jan-2
        #   Jan-2 day label covers hours 12:00 Jan-2 through 11:00 Jan-3 (not present)
        # Hours 0:00-11:00 Jan-1 → shifted label = Dec-31 → no daily value → NaN
        # Hours 12:00-23:00 Jan-1 → label = Jan-1 (12 hours)
        # Hours 0:00-11:00 Jan-2 → label = Jan-1 (12 hours)
        # Hours 12:00-23:00 Jan-2 → label = Jan-2 (12 hours)
        jan1_hours = result["2000-01-01 12":"2000-01-02 11"]
        assert abs(jan1_hours.sum() - 10.0) < 1e-10
        jan2_hours = result["2000-01-02 12":"2000-01-02 23"]
        assert abs(jan2_hours.sum() - 20.0) < 1e-10


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


# ---------------------------------------------------------------------------
# _day_labels
# ---------------------------------------------------------------------------


class TestDayLabels:
    def test_default_midnight(self):
        from met_timeseries.disaggregation import _day_labels
        import datetime

        idx = pd.date_range("2000-01-01 22:00", periods=4, freq="h")
        labels = _day_labels(idx, day_start_hour=0)
        assert labels[0] == datetime.date(2000, 1, 1)
        assert labels[1] == datetime.date(2000, 1, 1)
        assert labels[2] == datetime.date(2000, 1, 2)  # 00:00 Jan 2
        assert labels[3] == datetime.date(2000, 1, 2)

    def test_midday_boundary(self):
        from met_timeseries.disaggregation import _day_labels
        import datetime

        # 11:00 and 12:00 on Jan 2
        idx = pd.date_range("2000-01-02 11:00", periods=2, freq="h")
        labels = _day_labels(idx, day_start_hour=12)
        # 11:00 - 12h = 23:00 Jan 1 → date = Jan 1
        assert labels[0] == datetime.date(2000, 1, 1)
        # 12:00 - 12h = 00:00 Jan 2 → date = Jan 2
        assert labels[1] == datetime.date(2000, 1, 2)


# ---------------------------------------------------------------------------
# _cascade_halve / cascade_disaggregate
# ---------------------------------------------------------------------------


class TestCascadeHalve:
    def test_mass_conservation(self):
        from met_timeseries.disaggregation import _cascade_halve

        rf = np.array([10.0, 0.0, 5.0, 0.0, 20.0])
        result = _cascade_halve(rf, tau_obs=0.5, A=1.0, lp=0.5)
        assert len(result) == 10
        # Total mass must be preserved
        np.testing.assert_allclose(result.sum(), rf.sum(), rtol=1e-10)

    def test_zero_input_stays_zero(self):
        from met_timeseries.disaggregation import _cascade_halve

        rf = np.array([0.0, 0.0, 0.0])
        result = _cascade_halve(rf, tau_obs=0.5, A=1.0, lp=0.5)
        np.testing.assert_allclose(result, 0.0)


class TestCascadeDisaggregate:
    def test_name_and_frequency(self):
        from met_timeseries.disaggregation import cascade_disaggregate

        daily = _daily_series([5.0, 10.0, 3.0])
        result = cascade_disaggregate(daily, tau_obs=0.5, A=1.0, lp=0.5)
        assert result.name == "precip_mm"
        assert result.index.freq == "h" or pd.infer_freq(result.index) == "h"

    def test_mass_conservation(self):
        from met_timeseries.disaggregation import cascade_disaggregate

        daily = _daily_series([5.0, 0.0, 10.0, 8.0, 0.0])
        result = cascade_disaggregate(daily, tau_obs=0.5, A=1.0, lp=0.5)
        # Allow small tolerance from Güntner resampling truncation
        np.testing.assert_allclose(result.sum(), daily.sum(), rtol=0.01)

    def test_aligned_to_start(self):
        """Cascade output should start at the same timestamp as daily input."""
        from met_timeseries.disaggregation import cascade_disaggregate

        daily = _daily_series([5.0, 10.0], start="2000-03-15")
        result = cascade_disaggregate(daily, tau_obs=0.5, A=1.0, lp=0.5)
        assert result.index[0] == pd.Timestamp("2000-03-15 00:00")

    def test_zero_daily_produces_zero_hourly(self):
        from met_timeseries.disaggregation import cascade_disaggregate

        daily = _daily_series([0.0, 0.0, 0.0])
        result = cascade_disaggregate(daily, tau_obs=0.5, A=1.0, lp=0.5)
        np.testing.assert_allclose(result.values, 0.0)
