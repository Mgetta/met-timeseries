"""Tests for FAO-56 PET derivation."""

from __future__ import annotations

import numpy as np
import pytest

from met_timeseries.derivations.pet import compute_pet_fao56


def test_pet_positive():
    """ET should be non-negative."""
    et = compute_pet_fao56(
        temp_k=np.array([300.0]),
        dewpoint_k=np.array([285.0]),
        wind_speed_2m=np.array([2.0]),
        solar_rad=np.array([600.0]),
        pressure=np.array([101325.0]),
    )
    assert float(et[0]) >= 0.0


def test_pet_zero_radiation():
    """At night (zero solar radiation) ET should be near zero or small."""
    et = compute_pet_fao56(
        temp_k=np.array([290.0]),
        dewpoint_k=np.array([280.0]),
        wind_speed_2m=np.array([2.0]),
        solar_rad=np.array([0.0]),
        pressure=np.array([101325.0]),
    )
    assert float(et[0]) >= 0.0


def test_pet_increases_with_temperature():
    """Higher temperature should yield higher ET (all else equal)."""
    et_low = compute_pet_fao56(
        temp_k=np.array([280.0]),
        dewpoint_k=np.array([270.0]),
        wind_speed_2m=np.array([2.0]),
        solar_rad=np.array([500.0]),
        pressure=np.array([101325.0]),
    )
    et_high = compute_pet_fao56(
        temp_k=np.array([310.0]),
        dewpoint_k=np.array([270.0]),
        wind_speed_2m=np.array([2.0]),
        solar_rad=np.array([500.0]),
        pressure=np.array([101325.0]),
    )
    assert float(et_high[0]) >= float(et_low[0])


def test_pet_xarray():
    import xarray as xr

    et = compute_pet_fao56(
        temp_k=xr.DataArray([300.0], dims=["time"]),
        dewpoint_k=xr.DataArray([285.0], dims=["time"]),
        wind_speed_2m=xr.DataArray([2.0], dims=["time"]),
        solar_rad=xr.DataArray([600.0], dims=["time"]),
        pressure=xr.DataArray([101325.0], dims=["time"]),
    )
    assert et.name == "pet"
    assert et.attrs["units"] == "mm/hr"
    assert float(et.values[0]) >= 0.0


def test_pet_array_shape():
    shape = (24, 10, 10)
    et = compute_pet_fao56(
        temp_k=np.full(shape, 295.0),
        dewpoint_k=np.full(shape, 280.0),
        wind_speed_2m=np.full(shape, 3.0),
        solar_rad=np.full(shape, 400.0),
        pressure=np.full(shape, 101325.0),
    )
    assert et.shape == shape
