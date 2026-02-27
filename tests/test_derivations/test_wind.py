"""Tests for wind speed derivation."""

from __future__ import annotations

import numpy as np
import pytest

from met_timeseries.derivations.wind import compute_wind_speed


def test_wind_speed_pythagorean():
    u = np.array([3.0])
    v = np.array([4.0])
    result = compute_wind_speed(u, v)
    np.testing.assert_allclose(result, [5.0])


def test_wind_speed_zero():
    result = compute_wind_speed(np.array([0.0]), np.array([0.0]))
    np.testing.assert_allclose(result, [0.0])


def test_wind_speed_negative_components():
    u = np.array([-3.0])
    v = np.array([-4.0])
    result = compute_wind_speed(u, v)
    np.testing.assert_allclose(result, [5.0])


def test_wind_speed_2d():
    u = np.ones((4, 4)) * 3.0
    v = np.ones((4, 4)) * 4.0
    result = compute_wind_speed(u, v)
    np.testing.assert_allclose(result, np.full((4, 4), 5.0))


def test_wind_speed_xarray():
    import xarray as xr

    u = xr.DataArray([3.0, 0.0], dims=["time"])
    v = xr.DataArray([4.0, 5.0], dims=["time"])
    result = compute_wind_speed(u, v)
    assert result.name == "wind_speed"
    assert result.attrs["units"] == "m/s"
    np.testing.assert_allclose(result.values, [5.0, 5.0])
