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
