"""Tests for cloud cover derivation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from met_timeseries.derivations.cloud_cover import compute_cloud_cover


def _make_dswrf(values, times, lats, lons=None):
    """Helper to create a dswrf DataArray."""
    if lons is None:
        lons = [-84.0]
    return xr.DataArray(
        np.array(values, dtype=float).reshape(len(times), len(lats), len(lons)),
        coords={"time": np.array(times, dtype="datetime64[s]"), "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
    )


def test_zero_radiation_at_noon_gives_high_cloud_cover():
    times = ["2020-06-21T12:00:00"]
    lats = [35.0]
    dswrf = _make_dswrf([0.0], times, lats)
    cc = compute_cloud_cover(dswrf)
    # SW=0 during daytime -> cloud cover=1.0
    assert float(cc.values[0, 0, 0]) == pytest.approx(1.0)


def test_high_radiation_gives_low_cloud_cover():
    # Provide radiation equal to clear-sky estimate (should give ~0 cloud cover)
    times = ["2020-06-21T12:00:00"]
    lats = [35.0]
    # At lat=35, June 21, cos(zenith) approx 0.9 -> sw_clear approx 1367 * 0.9 approx 1230
    dswrf = _make_dswrf([1230.0], times, lats)
    cc = compute_cloud_cover(dswrf)
    val = float(cc.values[0, 0, 0])
    assert 0.0 <= val <= 0.3  # low cloud cover expected


def test_night_time_is_nan():
    times = ["2020-06-21T00:00:00"]
    lats = [35.0]
    dswrf = _make_dswrf([0.0], times, lats)
    cc = compute_cloud_cover(dswrf)
    assert np.isnan(float(cc.values[0, 0, 0]))


def test_output_shape_preserved(sample_gdf):
    times = ["2020-06-21T12:00:00", "2020-06-21T13:00:00"]
    lats = [35.0, 35.5]
    lons = [-84.0, -83.5]
    dswrf = xr.DataArray(
        np.ones((2, 2, 2)) * 500.0,
        coords={"time": np.array(times, dtype="datetime64[s]"), "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
    )
    cc = compute_cloud_cover(dswrf)
    assert cc.shape == (2, 2, 2)
