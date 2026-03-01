"""
Tests for derive_variables() in met_timeseries.derivations.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

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
