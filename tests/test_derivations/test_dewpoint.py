"""Tests for dewpoint temperature derivation."""

from __future__ import annotations

import numpy as np
import pytest

from met_timeseries.derivations.dewpoint import compute_dewpoint


def test_dewpoint_below_air_temperature():
    # At T=300K, q=0.01 kg/kg, P=101325 Pa, dewpoint should be < T
    q = np.array([0.01])
    P = np.array([101325.0])
    T = np.array([300.0])
    td = compute_dewpoint(q, P, T)
    assert float(td[0]) < 300.0


def test_dewpoint_near_saturation():
    # Very high specific humidity -> dewpoint approx air temperature
    q = np.array([0.030])
    P = np.array([101325.0])
    T = np.array([305.0])
    td = compute_dewpoint(q, P, T)
    # Dewpoint should be close to (or equal to) T, clipped
    assert float(td[0]) <= float(T[0])


def test_dewpoint_array():
    q = np.array([0.005, 0.010, 0.015])
    P = np.array([101325.0, 101325.0, 101325.0])
    T = np.array([290.0, 295.0, 300.0])
    td = compute_dewpoint(q, P, T)
    assert td.shape == (3,)
    assert all(td[i] <= T[i] for i in range(3))


def test_dewpoint_xarray():
    import xarray as xr

    q = xr.DataArray([0.01], dims=["time"])
    P = xr.DataArray([101325.0], dims=["time"])
    T = xr.DataArray([300.0], dims=["time"])
    td = compute_dewpoint(q, P, T)
    assert td.name == "dewpoint"
    assert td.attrs["units"] == "K"
