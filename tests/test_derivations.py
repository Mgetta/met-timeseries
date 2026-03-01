"""Tests for met_timeseries.derivations."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from met_timeseries.derivations import derive_variables


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
