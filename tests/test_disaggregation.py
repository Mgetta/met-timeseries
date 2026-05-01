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


# ---------------------------------------------------------------------------
# xarray helpers
# ---------------------------------------------------------------------------


def _daily_da(values, start="2000-01-01"):
    """Create a daily xr.DataArray from a list of values.

    Note: pass at least 3 values if the result will be passed to
    functions that call ``_infer_freq`` (``pd.infer_freq`` requires
    at least 3 timestamps for unambiguous frequency detection).
    """
    import xarray as xr
    index = pd.date_range(start, periods=len(values), freq="D")
    return xr.DataArray(np.array(values, dtype=float), coords={"time": index}, dims=["time"])


def _hourly_da(values, start="2000-01-01"):
    """Create an hourly xr.DataArray from a list of values."""
    import xarray as xr
    index = pd.date_range(start, periods=len(values), freq="h")
    return xr.DataArray(np.array(values, dtype=float), coords={"time": index}, dims=["time"])


def _paired(daily_values, hourly_values_per_day, start="2000-01-01"):
    """Return aligned (coarse, fine) pair spanning len(daily_values) days."""
    n = len(daily_values)
    coarse = _daily_da(daily_values, start=start)
    flat = list(hourly_values_per_day) * n
    fine = _hourly_da(flat, start=start)
    return coarse, fine


# ---------------------------------------------------------------------------
# _infer_freq
# ---------------------------------------------------------------------------


class TestInferFreq:
    def test_daily(self):
        from met_timeseries.disaggregation import _infer_freq
        da = _daily_da([1.0, 2.0, 3.0])
        freq = _infer_freq(da)
        assert freq is not None

    def test_hourly(self):
        from met_timeseries.disaggregation import _infer_freq
        da = _hourly_da([1.0] * 24)
        freq = _infer_freq(da)
        assert freq is not None

    def test_raises_on_irregular(self):
        import xarray as xr
        from met_timeseries.disaggregation import _infer_freq
        times = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"])
        da = xr.DataArray([1.0, 2.0, 3.0], coords={"time": times}, dims=["time"])
        with pytest.raises(ValueError, match="Cannot infer time frequency"):
            _infer_freq(da)

    def test_error_message_includes_name_and_timestamp(self):
        import xarray as xr
        from met_timeseries.disaggregation import _infer_freq
        times = pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-04"])
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            coords={"time": times},
            dims=["time"],
            name="precip",
        )
        with pytest.raises(ValueError) as exc_info:
            _infer_freq(da)
        msg = str(exc_info.value)
        assert "precip" in msg
        assert "2000-01-01" in msg


# ---------------------------------------------------------------------------
# _count_fine_per_coarse
# ---------------------------------------------------------------------------


class TestCountFinePerCoarse:
    def test_24_hours_per_day(self):
        from met_timeseries.disaggregation import _count_fine_per_coarse
        fine = _hourly_da([1.0] * 72)  # 3 days
        counts = _count_fine_per_coarse(fine, "1D")
        assert float(counts.isel(time=0)) == 24.0
        assert float(counts.isel(time=23)) == 24.0


# ---------------------------------------------------------------------------
# _normalise_weights
# ---------------------------------------------------------------------------


class TestNormaliseWeights:
    def test_sums_to_one_per_day(self):
        from met_timeseries.disaggregation import _normalise_weights
        # 3 days of a repeating 24-h pattern
        pattern = _hourly_da([1.0, 2.0, 3.0, 4.0] * 6 + [1.0, 2.0, 3.0, 4.0] * 6 + [1.0, 2.0, 3.0, 4.0] * 6)
        weights = _normalise_weights(pattern, "1D")
        # Each day's weights should sum to 1.0
        day0_sum = float(weights.isel(time=slice(0, 24)).sum())
        assert abs(day0_sum - 1.0) < 1e-10

    def test_uniform_fallback_zero_pattern(self):
        from met_timeseries.disaggregation import _normalise_weights
        pattern = _hourly_da([0.0] * 72)  # 3 days of zeros
        weights = _normalise_weights(pattern, "1D")
        np.testing.assert_allclose(weights.values, 1.0 / 24, rtol=1e-6)


# ---------------------------------------------------------------------------
# _disaggregate_sum
# ---------------------------------------------------------------------------


class TestDisaggregateSum:
    def test_conservation(self):
        from met_timeseries.disaggregation import _normalise_weights, _disaggregate_sum
        coarse, fine = _paired([12.0, 12.0, 12.0], [1.0] * 24)
        weights = _normalise_weights(fine, "1D")
        result = _disaggregate_sum(coarse, weights)
        # Each day's sum should equal 12.0 (total = 36.0)
        assert abs(float(result.sum()) - 36.0) < 1e-9

    def test_no_negatives(self):
        from met_timeseries.disaggregation import _normalise_weights, _disaggregate_sum
        coarse, fine = _paired([5.0, 5.0, 5.0], [0.5] * 24)
        weights = _normalise_weights(fine, "1D")
        result = _disaggregate_sum(coarse, weights)
        assert float(result.min()) >= 0.0


# ---------------------------------------------------------------------------
# _disaggregate_mean_additive
# ---------------------------------------------------------------------------


class TestDisaggregateMeanAdditive:
    def test_mean_preserved(self):
        from met_timeseries.disaggregation import _disaggregate_mean_additive
        coarse, fine = _paired([15.0, 15.0, 15.0], list(range(24)))
        result = _disaggregate_mean_additive(coarse, fine)
        # First day mean should equal 15.0
        assert abs(float(result.isel(time=slice(0, 24)).mean()) - 15.0) < 1e-9

    def test_abs_differences_preserved(self):
        from met_timeseries.disaggregation import _disaggregate_mean_additive
        coarse, fine = _paired([10.0, 10.0, 10.0], [0.0, 1.0, 2.0, 3.0] * 6)
        result = _disaggregate_mean_additive(coarse, fine)
        result_vals = result.values
        fine_vals = fine.values
        np.testing.assert_allclose(
            result_vals[1:] - result_vals[:-1],
            fine_vals[1:] - fine_vals[:-1],
            atol=1e-9,
        )


# ---------------------------------------------------------------------------
# _disaggregate_mean_multiplicative
# ---------------------------------------------------------------------------


class TestDisaggregateMeanMultiplicative:
    def test_mean_preserved(self):
        from met_timeseries.disaggregation import _disaggregate_mean_multiplicative
        coarse, fine = _paired([6.0, 6.0, 6.0], [2.0, 4.0, 6.0, 8.0] * 6)  # mean=5
        result = _disaggregate_mean_multiplicative(coarse, fine)
        assert abs(float(result.isel(time=slice(0, 24)).mean()) - 6.0) < 1e-9

    def test_no_negatives(self):
        from met_timeseries.disaggregation import _disaggregate_mean_multiplicative
        coarse, fine = _paired([0.0, 0.0, 0.0], [1.0, 2.0] * 12)
        result = _disaggregate_mean_multiplicative(coarse, fine)
        assert float(result.min()) >= 0.0


# ---------------------------------------------------------------------------
# disaggregate() primary API
# ---------------------------------------------------------------------------


class TestDisaggregateAPI:
    def test_sum_conservation(self):
        from met_timeseries.disaggregation import disaggregate
        coarse, fine = _paired([10.0, 10.0, 10.0], [1.0] * 24)
        result = disaggregate(coarse, conservation="sum", fine_pattern=fine)
        assert abs(float(result.sum()) - 30.0) < 1e-9

    def test_mean_additive_mean_preserved(self):
        from met_timeseries.disaggregation import disaggregate
        coarse, fine = _paired([20.0, 20.0, 20.0], list(range(24)))
        result = disaggregate(coarse, conservation="mean_additive", fine_pattern=fine)
        assert abs(float(result.isel(time=slice(0, 24)).mean()) - 20.0) < 1e-9

    def test_mean_multiplicative_mean_preserved(self):
        from met_timeseries.disaggregation import disaggregate
        coarse, fine = _paired([8.0, 8.0, 8.0], [2.0, 4.0, 6.0, 8.0] * 6)  # mean=5
        result = disaggregate(coarse, conservation="mean_multiplicative", fine_pattern=fine)
        assert abs(float(result.isel(time=slice(0, 24)).mean()) - 8.0) < 1e-9

    def test_sum_requires_fine_pattern_proportional(self):
        from met_timeseries.disaggregation import disaggregate
        # Validation happens before freq inference, so a single-value DataArray is fine.
        coarse = _daily_da([5.0])
        with pytest.raises(ValueError, match="fine_pattern"):
            disaggregate(coarse, conservation="sum")

    def test_mean_additive_requires_fine_pattern(self):
        from met_timeseries.disaggregation import disaggregate
        coarse = _daily_da([5.0])
        with pytest.raises(ValueError, match="fine_pattern"):
            disaggregate(coarse, conservation="mean_additive")

    def test_mean_multiplicative_requires_fine_pattern(self):
        from met_timeseries.disaggregation import disaggregate
        coarse = _daily_da([5.0])
        with pytest.raises(ValueError, match="fine_pattern"):
            disaggregate(coarse, conservation="mean_multiplicative")

    def test_unknown_conservation_raises(self):
        from met_timeseries.disaggregation import disaggregate
        coarse = _daily_da([5.0])
        with pytest.raises(ValueError, match="Unknown conservation"):
            disaggregate(coarse, conservation="bogus")  # type: ignore[arg-type]

    def test_custom_weights(self):
        from met_timeseries.disaggregation import disaggregate
        import xarray as xr
        coarse = _daily_da([24.0, 24.0, 24.0])
        # uniform weights: each hour gets 1/24
        times = pd.date_range("2000-01-01", periods=72, freq="h")
        weights = xr.DataArray(
            np.full(72, 1 / 24.0), coords={"time": times}, dims=["time"]
        )
        result = disaggregate(coarse, conservation="sum", weight_method="custom", weights=weights)
        np.testing.assert_allclose(result.values, 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# Named xarray convenience wrappers
# ---------------------------------------------------------------------------


class TestXarrayConvenienceWrappers:
    def test_disaggregate_radiation_sums(self):
        from met_timeseries.disaggregation import disaggregate_radiation
        coarse, fine = _paired([200.0, 200.0, 200.0], [1.0] * 24)
        result = disaggregate_radiation(coarse, fine)
        assert abs(float(result.sum()) - 600.0) < 1e-9

    def test_disaggregate_temperature_mean(self):
        from met_timeseries.disaggregation import disaggregate_temperature
        coarse, fine = _paired([15.0, 15.0, 15.0], list(range(24)))
        result = disaggregate_temperature(coarse, fine)
        assert abs(float(result.isel(time=slice(0, 24)).mean()) - 15.0) < 1e-9

    def test_disaggregate_dewpoint_mean(self):
        from met_timeseries.disaggregation import disaggregate_dewpoint
        coarse, fine = _paired([5.0, 5.0, 5.0], [3.0] * 24)
        result = disaggregate_dewpoint(coarse, fine)
        assert abs(float(result.isel(time=slice(0, 24)).mean()) - 5.0) < 1e-9

    def test_disaggregate_wind_mean(self):
        from met_timeseries.disaggregation import disaggregate_wind
        coarse, fine = _paired([6.0, 6.0, 6.0], [2.0, 4.0, 6.0, 8.0] * 6)  # mean=5
        result = disaggregate_wind(coarse, fine)
        assert abs(float(result.isel(time=slice(0, 24)).mean()) - 6.0) < 1e-9

    def test_disaggregate_precipitation_xarray_dispatch(self):
        from met_timeseries.disaggregation import disaggregate_precipitation
        coarse, fine = _paired([10.0, 10.0, 10.0], [1.0] * 24)
        result = disaggregate_precipitation(coarse, fine)
        assert abs(float(result.sum()) - 30.0) < 1e-9

    def test_disaggregate_precipitation_pandas_dispatch(self):
        from met_timeseries.disaggregation import disaggregate_precipitation
        # pandas path preserved (uses proportional, not cascade, for determinism)
        daily = _daily_series([10.0])
        pattern = _hourly_series([1.0] * 24)
        result = disaggregate_precipitation(daily, pattern, method="proportional")
        assert result.name == "precip_mm"
        assert abs(result.sum() - 10.0) < 1e-9
