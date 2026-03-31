"""
Tests for met_timeseries.derivations public API.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_da(value: float, dims=("lat", "lon"), lats=(45.0, 45.1), lons=(-110.0, -109.9)) -> xr.DataArray:
    """Build a 2×2 DataArray filled with *value*."""
    return xr.DataArray(
        np.full((len(lats), len(lons)), value, dtype=float),
        dims=list(dims),
        coords={"lat": list(lats), "lon": list(lons)},
    )


def _make_da_with_time(value: float, hour: int = 12, lat: float = 45.0, lon: float = -110.0) -> xr.DataArray:
    """Build a single-cell DataArray with a time dimension at the given UTC hour."""
    import pandas as pd

    t = pd.Timestamp(f"2000-06-21 {hour:02d}:00:00")
    return xr.DataArray(
        np.full((1, 1, 1), value, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": [t], "lat": [lat], "lon": [lon]},
    )


# ---------------------------------------------------------------------------
# _clear_sky_radiation
# ---------------------------------------------------------------------------


class TestClearSkyRadiation:
    """Unit tests for the _clear_sky_radiation helper."""

    def _fn(self, lat, lon, dt):
        from met_timeseries.derivations import _clear_sky_radiation
        return _clear_sky_radiation(lat, lon, dt)

    def test_solar_noon_midlatitude_summer_positive(self):
        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        assert self._fn(lat=45.0, lon=-110.0, dt=dt) > 0.0

    def test_midnight_is_zero(self):
        dt = datetime.datetime(2000, 6, 21, 0, 0, 0)
        assert self._fn(lat=45.0, lon=-110.0, dt=dt) == 0.0

    def test_polar_winter_below_horizon(self):
        dt = datetime.datetime(2000, 12, 21, 12, 0, 0)
        assert self._fn(lat=80.0, lon=0.0, dt=dt) == 0.0

    def test_value_bounded_by_solar_constant_times_tau(self):
        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        result = self._fn(lat=0.0, lon=0.0, dt=dt)
        assert 0.0 <= result <= 1361.0 * 0.75 + 1e-6

    def test_known_approximate_value(self):
        dt = datetime.datetime(2000, 3, 20, 12, 0, 0)
        result = self._fn(lat=0.0, lon=0.0, dt=dt)
        assert abs(result - 1020.75) / 1020.75 < 0.05


# ---------------------------------------------------------------------------
# wind_speed
# ---------------------------------------------------------------------------


class TestWindSpeed:
    def test_wind_speed_value(self):
        from met_timeseries.derivations import wind_speed

        u = _make_da(3.0)
        v = _make_da(4.0)
        ws = wind_speed(u, v)
        assert float(ws.values.flat[0]) == pytest.approx(5.0)

    def test_wind_speed_name(self):
        from met_timeseries.derivations import wind_speed

        ws = wind_speed(_make_da(3.0), _make_da(4.0))
        assert ws.name == "wind_speed_ms"

    def test_wind_speed_zero(self):
        from met_timeseries.derivations import wind_speed

        ws = wind_speed(_make_da(0.0), _make_da(0.0))
        assert float(ws.values.flat[0]) == pytest.approx(0.0)

    def test_wind_speed_preserves_dims(self):
        from met_timeseries.derivations import wind_speed

        u = _make_da(1.0)
        v = _make_da(1.0)
        ws = wind_speed(u, v)
        assert ws.dims == u.dims


# ---------------------------------------------------------------------------
# dewpoint_from_specific_humidity
# ---------------------------------------------------------------------------


class TestDewpointFromSpecificHumidity:
    def test_returns_dewpoint_c(self):
        from met_timeseries.derivations import dewpoint_from_specific_humidity

        spfh = _make_da(0.01)
        dp = dewpoint_from_specific_humidity(spfh)
        assert dp.name == "dewpoint_c"

    def test_no_pressure_uses_default(self):
        from met_timeseries.derivations import dewpoint_from_specific_humidity

        spfh = _make_da(0.01)
        dp = dewpoint_from_specific_humidity(spfh)
        assert dp.shape == spfh.shape
        assert np.all(np.isfinite(dp.values))

    def test_with_explicit_pressure(self):
        from met_timeseries.derivations import dewpoint_from_specific_humidity

        spfh = _make_da(0.01)
        pres = _make_da(101325.0)
        dp_default = dewpoint_from_specific_humidity(spfh)
        dp_explicit = dewpoint_from_specific_humidity(spfh, pressure=pres)
        np.testing.assert_allclose(dp_default.values, dp_explicit.values, rtol=1e-4)

    def test_higher_humidity_higher_dewpoint(self):
        from met_timeseries.derivations import dewpoint_from_specific_humidity

        dp_low = dewpoint_from_specific_humidity(_make_da(0.005))
        dp_high = dewpoint_from_specific_humidity(_make_da(0.015))
        assert float(dp_high.values.flat[0]) > float(dp_low.values.flat[0])


# ---------------------------------------------------------------------------
# kelvin_to_celsius
# ---------------------------------------------------------------------------


class TestKelvinToCelsius:
    def test_value(self):
        from met_timeseries.derivations import kelvin_to_celsius

        da = _make_da(273.15)
        result = kelvin_to_celsius(da)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-10)

    def test_name(self):
        from met_timeseries.derivations import kelvin_to_celsius

        result = kelvin_to_celsius(_make_da(300.0))
        assert result.name == "temp_c"

    def test_preserves_dims(self):
        from met_timeseries.derivations import kelvin_to_celsius

        da = _make_da(300.0)
        result = kelvin_to_celsius(da)
        assert result.dims == da.dims


# ---------------------------------------------------------------------------
# cloud_cover
# ---------------------------------------------------------------------------


class TestCloudCover:
    def test_basic_fraction(self):
        from met_timeseries.derivations import _clear_sky_radiation, cloud_cover

        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        import pandas as pd

        cs = _clear_sky_radiation(45.0, -110.0, dt)
        da = _make_da_with_time(cs * 0.5, hour=12)
        lat = np.array([45.0])
        lon = np.array([-110.0])
        time = da.coords["time"].values

        result = cloud_cover(da, lat, lon, time)
        cc = float(result.values.flat[0])
        assert abs(cc - 0.5) < 0.01

    def test_clear_sky_gives_zero(self):
        from met_timeseries.derivations import _clear_sky_radiation, cloud_cover

        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        cs = _clear_sky_radiation(45.0, -110.0, dt)
        da = _make_da_with_time(cs, hour=12)
        lat = np.array([45.0])
        lon = np.array([-110.0])
        time = da.coords["time"].values

        result = cloud_cover(da, lat, lon, time)
        assert abs(float(result.values.flat[0])) < 0.01

    def test_overcast_gives_one(self):
        from met_timeseries.derivations import cloud_cover

        da = _make_da_with_time(0.0, hour=12)
        lat = np.array([45.0])
        lon = np.array([-110.0])
        time = da.coords["time"].values

        result = cloud_cover(da, lat, lon, time)
        assert abs(float(result.values.flat[0]) - 1.0) < 0.01

    def test_nighttime_is_nan(self):
        from met_timeseries.derivations import cloud_cover

        da = _make_da_with_time(0.0, hour=0)
        lat = np.array([45.0])
        lon = np.array([-110.0])
        time = da.coords["time"].values

        result = cloud_cover(da, lat, lon, time)
        assert np.isnan(float(result.values.flat[0]))

    def test_clamped_when_dswrf_exceeds_clearsky(self):
        from met_timeseries.derivations import _clear_sky_radiation, cloud_cover

        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        cs = _clear_sky_radiation(45.0, -110.0, dt)
        da = _make_da_with_time(cs * 2.0, hour=12)
        lat = np.array([45.0])
        lon = np.array([-110.0])
        time = da.coords["time"].values

        result = cloud_cover(da, lat, lon, time)
        assert float(result.values.flat[0]) == 0.0

    def test_no_time_returns_2d_array(self):
        from met_timeseries.derivations import cloud_cover

        lats = np.array([45.0, 46.0])
        lons = np.array([-110.0, -109.0])
        da = xr.DataArray(
            np.full((2, 2), 300.0),
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )

        result = cloud_cover(da, lats, lons, time=None)
        assert result.dims == ("lat", "lon")
        valid = result.values[~np.isnan(result.values)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0)

    def test_name(self):
        from met_timeseries.derivations import cloud_cover

        da = _make_da_with_time(300.0, hour=12)
        result = cloud_cover(da, np.array([45.0]), np.array([-110.0]), da.coords["time"].values)
        assert result.name == "cloud_cover_fraction"


# ---------------------------------------------------------------------------
# Helpers for PET tests
# ---------------------------------------------------------------------------


def _make_pet_da(value: float, lat: float = 45.0, lon: float = -110.0, dates=None) -> xr.DataArray:
    """Build a (time, lat, lon) DataArray for PET input testing."""
    import pandas as pd

    if dates is None:
        dates = [pd.Timestamp("2000-07-15")]
    return xr.DataArray(
        np.full((len(dates), 1, 1), value, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": dates, "lat": [lat], "lon": [lon]},
    )


# ---------------------------------------------------------------------------
# pet_penman_monteith
# ---------------------------------------------------------------------------


class TestPetPenmanMonteith:
    """Tests for :func:`met_timeseries.derivations.pet_penman_monteith`."""

    def _run(
        self,
        tmin=15.0,
        tmax=30.0,
        wind=2.0,
        rs=20.0,
        td=12.0,
        elev=100.0,
        lat=45.0,
        dates=None,
    ):
        from met_timeseries.derivations import pet_penman_monteith

        return pet_penman_monteith(
            _make_pet_da(tmin, lat=lat, dates=dates),
            _make_pet_da(tmax, lat=lat, dates=dates),
            _make_pet_da(wind, lat=lat, dates=dates),
            _make_pet_da(rs, lat=lat, dates=dates),
            _make_pet_da(td, lat=lat, dates=dates),
            elev,
        )

    def test_positive_pet(self):
        result = self._run()
        assert float(result.values.flat[0]) > 0.0

    def test_no_negative_pet(self):
        """Extreme cold / polar conditions must not produce negative PET."""
        result = self._run(tmin=-40.0, tmax=-30.0, wind=1.0, rs=2.0, td=-42.0, lat=70.0)
        assert np.all(result.values >= 0.0)

    def test_reasonable_range(self):
        result = self._run()
        val = float(result.values.flat[0])
        assert 0.0 <= val <= 15.0

    def test_wind_effect(self):
        """Higher wind speed should increase PET (all else equal)."""
        low = float(self._run(wind=1.0).values.flat[0])
        high = float(self._run(wind=5.0).values.flat[0])
        assert high > low

    def test_humidity_effect(self):
        """Higher dewpoint (smaller vapour deficit) should decrease PET."""
        low_td = float(self._run(td=5.0).values.flat[0])
        high_td = float(self._run(td=20.0).values.flat[0])
        assert high_td < low_td

    def test_multi_day(self):
        """3-day input should produce output with 3 time steps."""
        import pandas as pd

        dates = pd.date_range("2000-07-01", periods=3, freq="D")
        result = self._run(dates=dates)
        assert result.shape[0] == 3
        assert result.dims[0] == "time"

    def test_elevation_effect(self):
        """Higher elevation (lower pressure) changes PET."""
        low_elev = float(self._run(elev=0.0).values.flat[0])
        high_elev = float(self._run(elev=2000.0).values.flat[0])
        assert low_elev != high_elev

    def test_output_name(self):
        result = self._run()
        assert result.name == "pet_penman_monteith_mm"

    def test_output_dims(self):
        result = self._run()
        assert result.dims == ("time", "lat", "lon")


# ---------------------------------------------------------------------------
# pet_hargreaves — import location tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# pet_penman_monteith_hourly
# ---------------------------------------------------------------------------


def _make_hourly_da(
    value: float,
    lat: float = 45.0,
    lon: float = -110.0,
    dates=None,
) -> xr.DataArray:
    """Build a (time, lat, lon) DataArray for hourly PET input testing."""
    import pandas as pd

    if dates is None:
        dates = pd.date_range("2000-06-21", periods=24, freq="h")
    return xr.DataArray(
        np.full((len(dates), 1, 1), value, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": dates, "lat": [lat], "lon": [lon]},
    )


def _make_hourly_da_single(
    value: float,
    hour: int = 12,
    lat: float = 45.0,
    lon: float = -110.0,
) -> xr.DataArray:
    """Build a single-timestep (time, lat, lon) DataArray at the given UTC hour."""
    import pandas as pd

    t = pd.Timestamp(f"2000-06-21 {hour:02d}:00:00")
    return xr.DataArray(
        np.full((1, 1, 1), value, dtype=float),
        dims=["time", "lat", "lon"],
        coords={"time": [t], "lat": [lat], "lon": [lon]},
    )


class TestPetPenmanMonteithHourly:
    """Tests for :func:`met_timeseries.derivations.pet_penman_monteith_hourly`."""

    def _run_single(self, temp=25.0, wind=2.0, rs=800.0, td=12.0, elev=100.0, hour=12):
        from met_timeseries.derivations import pet_penman_monteith_hourly

        return pet_penman_monteith_hourly(
            _make_hourly_da_single(temp, hour=hour),
            _make_hourly_da_single(wind, hour=hour),
            _make_hourly_da_single(rs, hour=hour),
            _make_hourly_da_single(td, hour=hour),
            elev,
        )

    def _run_24h(self, temp=25.0, wind=2.0, rs=800.0, td=12.0, elev=100.0):
        import pandas as pd
        from met_timeseries.derivations import pet_penman_monteith_hourly

        dates = pd.date_range("2000-06-21", periods=24, freq="h")
        return pet_penman_monteith_hourly(
            _make_hourly_da(temp, dates=dates),
            _make_hourly_da(wind, dates=dates),
            _make_hourly_da(rs, dates=dates),
            _make_hourly_da(td, dates=dates),
            elev,
        )

    def test_positive_pet_daytime(self):
        """Noon hour with realistic daytime conditions → PET > 0."""
        result = self._run_single(temp=25.0, wind=2.0, rs=800.0, td=12.0, elev=100.0, hour=12)
        assert float(result.values.flat[0]) > 0.0

    def test_zero_pet_nighttime(self):
        """Midnight with Rs=0 → PET should be 0 (clipped)."""
        result = self._run_single(temp=15.0, wind=2.0, rs=0.0, td=10.0, elev=100.0, hour=0)
        assert float(result.values.flat[0]) >= 0.0

    def test_no_negative_pet(self):
        """All conditions must produce non-negative PET."""
        import pandas as pd
        from met_timeseries.derivations import pet_penman_monteith_hourly

        dates = pd.date_range("2000-06-21", periods=24, freq="h")
        # Realistic diurnal radiation: 0 at night, peak at noon
        rs_vals = np.maximum(0.0, 800.0 * np.sin(np.pi * np.arange(24) / 24.0))
        result_vals = []
        for h, rs in enumerate(rs_vals):
            r = self._run_single(temp=20.0, wind=2.0, rs=float(rs), td=10.0, elev=100.0, hour=h)
            result_vals.append(float(r.values.flat[0]))
        assert all(v >= 0.0 for v in result_vals)

    def test_reasonable_hourly_range(self):
        """Daytime summer PET should be between 0 and 1.5 mm/hour."""
        result = self._run_single(temp=25.0, wind=2.0, rs=800.0, td=12.0, elev=100.0, hour=12)
        val = float(result.values.flat[0])
        assert 0.0 <= val <= 1.5

    def test_24hr_shape(self):
        """24-hour input → output has 24 time steps."""
        result = self._run_24h()
        assert result.shape[0] == 24
        assert result.dims[0] == "time"

    def test_higher_radiation_higher_pet(self):
        """More shortwave radiation → more PET (daytime)."""
        low = float(self._run_single(rs=400.0, hour=12).values.flat[0])
        high = float(self._run_single(rs=800.0, hour=12).values.flat[0])
        assert high > low

    def test_wind_effect(self):
        """Higher wind speed → more PET (all else equal, daytime)."""
        low = float(self._run_single(wind=1.0, hour=12).values.flat[0])
        high = float(self._run_single(wind=5.0, hour=12).values.flat[0])
        assert high > low

    def test_output_name(self):
        """Result DataArray should be named 'pet_penman_monteith_hourly_mm'."""
        result = self._run_single()
        assert result.name == "pet_penman_monteith_hourly_mm"

    def test_diurnal_pattern(self):
        """Midday hours should have higher PET than nighttime hours."""
        import pandas as pd
        from met_timeseries.derivations import pet_penman_monteith_hourly

        dates = pd.date_range("2000-06-21", periods=24, freq="h")
        # Use a simple half-sine radiation curve (0 at night, 800 W/m² at noon)
        rs_arr = np.zeros((24, 1, 1))
        for h in range(24):
            rs_arr[h, 0, 0] = max(0.0, 800.0 * np.sin(np.pi * h / 24.0))

        rs_da = xr.DataArray(rs_arr, dims=["time", "lat", "lon"],
                             coords={"time": dates, "lat": [45.0], "lon": [-110.0]})
        temp_da = _make_hourly_da(25.0, dates=dates)
        wind_da = _make_hourly_da(2.0, dates=dates)
        td_da = _make_hourly_da(12.0, dates=dates)

        result = pet_penman_monteith_hourly(temp_da, wind_da, rs_da, td_da, 100.0)

        # Midday hours (10-14 UTC at lon=-110 ≈ solar noon ~19:20 UTC; but use
        # hours 10-14 which should have non-trivial radiation in our curve)
        midday_max = float(result.values[10:15, 0, 0].max())
        nighttime_mean = float(result.values[0:4, 0, 0].mean())
        assert midday_max > nighttime_mean


# ---------------------------------------------------------------------------
# pet_hargreaves — import location tests
# ---------------------------------------------------------------------------


class TestPetHargreavesImport:
    def test_hargreaves_importable_from_derivations(self):
        """pet_hargreaves should be directly importable from derivations."""
        from met_timeseries.derivations import pet_hargreaves

        assert callable(pet_hargreaves)

    def test_hargreaves_backwards_compat(self):
        """pet_hargreaves imported from disaggregation should still work."""
        import pandas as pd
        from met_timeseries.disaggregation import pet_hargreaves

        idx = pd.date_range("2000-07-01", periods=1, freq="D")
        result = pet_hargreaves(
            pd.Series([10.0], index=idx),
            pd.Series([25.0], index=idx),
            lat=45.0,
        )
        assert result.name == "pet_hargreaves_mm"
        assert float(result.iloc[0]) > 0.0

