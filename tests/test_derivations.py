"""
Tests for derivations.py — clear-sky radiation helper and cloud-cover fraction.
Tests for derive_variables() in met_timeseries.derivations.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# _clear_sky_radiation tests
# ---------------------------------------------------------------------------

class TestClearSkyRadiation:
    """Unit tests for the _clear_sky_radiation helper."""

    def _fn(self, lat, lon, dt):
        from met_timeseries.derivations import _clear_sky_radiation
        return _clear_sky_radiation(lat, lon, dt)

    def test_solar_noon_midlatitude_summer_positive(self):
        """At solar noon in summer at a mid-latitude, clear-sky radiation should be positive."""
        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)  # summer solstice, noon UTC
        result = self._fn(lat=45.0, lon=-110.0, dt=dt)
        assert result > 0.0

    def test_midnight_is_zero(self):
        """At local midnight the sun is below the horizon — result must be 0."""
        dt = datetime.datetime(2000, 6, 21, 0, 0, 0)  # midnight UTC
        result = self._fn(lat=45.0, lon=-110.0, dt=dt)
        assert result == 0.0

    def test_polar_winter_below_horizon(self):
        """Near the poles in winter the sun stays below the horizon."""
        dt = datetime.datetime(2000, 12, 21, 12, 0, 0)  # winter solstice, noon UTC
        result = self._fn(lat=80.0, lon=0.0, dt=dt)
        assert result == 0.0

    def test_value_bounded_by_solar_constant_times_tau(self):
        """Clear-sky SW cannot exceed S₀ × τ (≈1020.75 W/m²)."""
        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        result = self._fn(lat=0.0, lon=0.0, dt=dt)  # equator at solstice noon
        assert 0.0 <= result <= 1361.0 * 0.75 + 1e-6  # allow tiny float error

    def test_known_approximate_value(self):
        """
        At the equator on the vernal equinox at solar noon (12 UTC ≈ solar noon),
        cos(zenith) ≈ 1.0 so R_clearsky ≈ 1361 × 0.75 ≈ 1020.75 W/m².
        """
        dt = datetime.datetime(2000, 3, 20, 12, 0, 0)  # vernal equinox, noon
        result = self._fn(lat=0.0, lon=0.0, dt=dt)
        # Accept within 5% tolerance (hour-angle approximation)
        assert abs(result - 1020.75) / 1020.75 < 0.05


# ---------------------------------------------------------------------------
# _cloud_cover / derive_variables cloud_cover_fraction tests
# ---------------------------------------------------------------------------

def _make_ds_with_dswrf(dswrf_value: float, lat=45.0, lon=-110.0, hour=12) -> xr.Dataset:
    """Return a minimal Dataset with a single-point DSWRF and a time coord at the given UTC hour."""
    import pandas as pd

    t = pd.Timestamp(f"2000-06-21 {hour:02d}:00:00")
    lats = np.array([lat])
    lons = np.array([lon])
    data = np.full((1, 1, 1), dswrf_value)  # (time, lat, lon)
    return xr.Dataset(
        {
            "DSWRF": xr.DataArray(
                data,
                dims=["time", "lat", "lon"],
                coords={"time": [t], "lat": lats, "lon": lons},
            )
        }
    )


class TestCloudCoverFraction:
    def test_basic_fraction(self):
        """If DSWRF = 300 and clear_sky ≈ 600, cloud_cover should be close to 0.5."""
        from met_timeseries.derivations import _clear_sky_radiation

        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        cs = _clear_sky_radiation(45.0, -110.0, dt)
        # Construct DSWRF that yields exactly 0.5 cloud cover
        dswrf_value = cs * 0.5
        ds = _make_ds_with_dswrf(dswrf_value, lat=45.0, lon=-110.0, hour=12)

        from met_timeseries.derivations import derive_variables
        derived = derive_variables(ds)

        assert "cloud_cover_fraction" in derived
        cc = float(derived["cloud_cover_fraction"].values.flat[0])
        assert abs(cc - 0.5) < 0.01

    def test_clear_sky_gives_zero(self):
        """When DSWRF equals clear-sky radiation exactly, cloud_cover should be 0."""
        from met_timeseries.derivations import _clear_sky_radiation

        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        cs = _clear_sky_radiation(45.0, -110.0, dt)
        ds = _make_ds_with_dswrf(cs, lat=45.0, lon=-110.0, hour=12)

        from met_timeseries.derivations import derive_variables
        derived = derive_variables(ds)

        cc = float(derived["cloud_cover_fraction"].values.flat[0])
        assert abs(cc) < 0.01

    def test_overcast_sky_gives_one(self):
        """When DSWRF is 0 and sun is up, cloud_cover should be 1."""
        ds = _make_ds_with_dswrf(0.0, lat=45.0, lon=-110.0, hour=12)

        from met_timeseries.derivations import derive_variables
        derived = derive_variables(ds)

        cc = float(derived["cloud_cover_fraction"].values.flat[0])
        assert abs(cc - 1.0) < 0.01

    def test_nighttime_is_nan(self):
        """Nighttime pixels (hour=0) should produce NaN cloud_cover."""
        ds = _make_ds_with_dswrf(0.0, lat=45.0, lon=-110.0, hour=0)

        from met_timeseries.derivations import derive_variables
        derived = derive_variables(ds)

        cc = float(derived["cloud_cover_fraction"].values.flat[0])
        assert np.isnan(cc)

    def test_clamped_when_dswrf_exceeds_clearsky(self):
        """If DSWRF > clear-sky (instrument artefact), cloud_cover must be clamped to 0."""
        from met_timeseries.derivations import _clear_sky_radiation

        dt = datetime.datetime(2000, 6, 21, 12, 0, 0)
        cs = _clear_sky_radiation(45.0, -110.0, dt)
        ds = _make_ds_with_dswrf(cs * 2.0, lat=45.0, lon=-110.0, hour=12)  # DSWRF >> clearsky

        from met_timeseries.derivations import derive_variables
        derived = derive_variables(ds)

        cc = float(derived["cloud_cover_fraction"].values.flat[0])
        assert cc == 0.0

    def test_no_dswrf_no_cloud_cover(self):
        """If DSWRF is absent, cloud_cover_fraction should not appear in derived dict."""
        ds = xr.Dataset(
            {
                "APCP": xr.DataArray(
                    np.ones((1, 1)),
                    dims=["lat", "lon"],
                    coords={"lat": [45.0], "lon": [-110.0]},
                )
            }
        )

        from met_timeseries.derivations import derive_variables
        derived = derive_variables(ds)

        assert "cloud_cover_fraction" not in derived

    def test_no_time_dimension_returns_array(self):
        """Datasets without a time dimension should still produce a cloud_cover_fraction array."""
        lats = np.array([45.0, 46.0])
        lons = np.array([-110.0, -109.0])
        data = np.full((2, 2), 300.0)
        ds = xr.Dataset(
            {
                "DSWRF": xr.DataArray(
                    data,
                    dims=["lat", "lon"],
                    coords={"lat": lats, "lon": lons},
                )
            }
        )

        from met_timeseries.derivations import derive_variables
        derived = derive_variables(ds)

        assert "cloud_cover_fraction" in derived
        cc = derived["cloud_cover_fraction"]
        assert cc.dims == ("lat", "lon")
        # Values should be either NaN (nighttime) or in [0, 1]
        valid = cc.values[~np.isnan(cc.values)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0)
from met_timeseries.derivations import derive_variables


def _make_dataset(**kwargs) -> xr.Dataset:
    """Build a minimal 2×2 Dataset from keyword-value pairs (scalar values)."""
    lats = np.array([45.0, 45.1])
    lons = np.array([-110.0, -109.9])
    dims = ["lat", "lon"]
    coords = {"lat": lats, "lon": lons}
    data_vars = {
        name: xr.DataArray(np.full((2, 2), val, dtype=float), dims=dims, coords=coords)
        for name, val in kwargs.items()
    }
    return xr.Dataset(data_vars)


class TestPetMm:
    def test_pet_mm_present_when_pevap_in_input(self) -> None:
        ds = _make_dataset(PEVAP=3.5)
        result = derive_variables(ds)
        assert "pet_mm" in result

    def test_pet_mm_value_equals_pevap(self) -> None:
        ds = _make_dataset(PEVAP=3.5)
        result = derive_variables(ds)
        np.testing.assert_allclose(result["pet_mm"].values, 3.5)

    def test_pet_mm_name(self) -> None:
        ds = _make_dataset(PEVAP=2.0)
        result = derive_variables(ds)
        assert result["pet_mm"].name == "pet_mm"

    def test_no_hargreaves_when_pevap_present(self) -> None:
        """PEVAP takes priority; Hargreaves fallback should NOT appear."""
        ds = _make_dataset(PEVAP=3.5, TMP=300.0, DSWRF=250.0)
        result = derive_variables(ds)
        assert "pet_hargreaves_mm" not in result
        assert "pet_mm" in result


class TestPetHargreaves:
    def test_hargreaves_present_when_pevap_absent(self) -> None:
        ds = _make_dataset(TMP=295.0, DSWRF=250.0)
        result = derive_variables(ds)
        assert "pet_hargreaves_mm" in result

    def test_no_hargreaves_without_dswrf(self) -> None:
        ds = _make_dataset(TMP=295.0)
        result = derive_variables(ds)
        assert "pet_hargreaves_mm" not in result

    def test_no_hargreaves_without_tmp(self) -> None:
        ds = _make_dataset(DSWRF=250.0)
        result = derive_variables(ds)
        assert "pet_hargreaves_mm" not in result

    def test_hargreaves_name(self) -> None:
        ds = _make_dataset(TMP=295.0, DSWRF=250.0)
        result = derive_variables(ds)
        assert result["pet_hargreaves_mm"].name == "pet_hargreaves_mm"

    def test_hargreaves_reasonable_values(self) -> None:
        """PET should be positive and in a plausible range for warm, sunny conditions."""
        # T ≈ 25 °C (298.15 K), DSWRF = 300 W/m²
        ds = _make_dataset(TMP=298.15, DSWRF=300.0)
        result = derive_variables(ds)
        pet = float(result["pet_hargreaves_mm"].values[0, 0])
        assert pet > 0, "PET should be positive"
        # Expected: 0.0023 * 300 * (25 + 17.8) * sqrt(10) ≈ 93.6
        expected = 0.0023 * 300.0 * (25.0 + 17.8) * np.sqrt(10.0)
        np.testing.assert_allclose(pet, expected, rtol=1e-5)

    def test_hargreaves_clipped_at_zero(self) -> None:
        """Negative PET values (very cold / dark) should be clipped to 0."""
        # T = -30 °C (243.15 K), DSWRF = 10 W/m²
        ds = _make_dataset(TMP=243.15, DSWRF=10.0)
        result = derive_variables(ds)
        pet = result["pet_hargreaves_mm"].values
        assert np.all(pet >= 0.0), "PET should never be negative"
def _make_ds(**kwargs) -> xr.Dataset:
    """Build a minimal single-cell Dataset from scalar keyword arguments."""
    return xr.Dataset(
        {
            name: xr.DataArray(np.array([[value]]), dims=["lat", "lon"])
            for name, value in kwargs.items()
        }
    )


class TestWindSpeed:
    def test_wind_speed_produced(self) -> None:
        ds = _make_ds(UGRD=3.0, VGRD=4.0)
        result = derive_variables(ds)
        assert "wind_speed_ms" in result

    def test_wind_speed_value(self) -> None:
        ds = _make_ds(UGRD=3.0, VGRD=4.0)
        result = derive_variables(ds)
        assert float(result["wind_speed_ms"].values.flat[0]) == pytest.approx(5.0)

    def test_wind_speed_missing_ugrd(self) -> None:
        ds = _make_ds(VGRD=4.0)
        result = derive_variables(ds)
        assert "wind_speed_ms" not in result

    def test_wind_speed_missing_vgrd(self) -> None:
        ds = _make_ds(UGRD=3.0)
        result = derive_variables(ds)
        assert "wind_speed_ms" not in result

    def test_wind_speed_zero(self) -> None:
        ds = _make_ds(UGRD=0.0, VGRD=0.0)
        result = derive_variables(ds)
        assert float(result["wind_speed_ms"].values.flat[0]) == pytest.approx(0.0)
