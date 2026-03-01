"""
Tests for met_timeseries.sources.nldas — earthaccess-based implementation.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture()
def bounds():
    from met_timeseries.sources.base import BoundingBox

    return BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)


def _make_mock_dataset(variables=("APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD")):
    """Create a small mock xarray Dataset mimicking NLDAS-2 output."""
    times = pd.date_range("2010-01-01", periods=2, freq="h")
    lats = np.array([44.9375, 45.0625, 45.9375, 46.0625])
    lons = np.array([-110.0625, -109.9375, -109.0625, -108.9375])

    data_vars = {
        var: (["time", "lat", "lon"], np.ones((2, 4, 4), dtype=np.float32))
        for var in variables
    }

    return xr.Dataset(
        data_vars,
        coords={"time": times, "lat": lats, "lon": lons},
    )


class TestFetchNldasCallsEarthaccess:
    """Verify that fetch_nldas calls earthaccess correctly."""

    @patch("met_timeseries.sources.nldas.xr.open_mfdataset")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_calls_login(self, mock_login, mock_search, mock_open, mock_mfdataset, bounds):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_mfdataset.return_value = _make_mock_dataset()

        from met_timeseries.sources.nldas import fetch_nldas

        fetch_nldas(bounds, 2010, 1)

        mock_login.assert_called_once()

    @patch("met_timeseries.sources.nldas.xr.open_mfdataset")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_calls_search_data_with_correct_params(
        self, mock_login, mock_search, mock_open, mock_mfdataset, bounds
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_mfdataset.return_value = _make_mock_dataset()

        from met_timeseries.sources.nldas import fetch_nldas

        fetch_nldas(bounds, 2010, 1)

        mock_search.assert_called_once_with(
            short_name="NLDAS_FORA0125_H",
            version="2.0",
            temporal=("2010-01-01", "2010-01-31"),
            bounding_box=(-110.0, 45.0, -109.0, 46.0),
        )

    @patch("met_timeseries.sources.nldas.xr.open_mfdataset")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_calls_open_with_search_results(
        self, mock_login, mock_search, mock_open, mock_mfdataset, bounds
    ):
        granules = [MagicMock()]
        mock_search.return_value = granules
        mock_open.return_value = [MagicMock()]
        mock_mfdataset.return_value = _make_mock_dataset()

        from met_timeseries.sources.nldas import fetch_nldas

        fetch_nldas(bounds, 2010, 1)

        mock_open.assert_called_once_with(granules)

    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_raises_when_no_granules_found(self, mock_login, mock_search, bounds):
        mock_search.return_value = []

        from met_timeseries.sources.nldas import fetch_nldas

        with pytest.raises(RuntimeError, match="No NLDAS-2 granules found"):
            fetch_nldas(bounds, 2010, 1)

    @patch("met_timeseries.sources.nldas.xr.open_mfdataset")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_dataset_has_requested_variables(
        self, mock_login, mock_search, mock_open, mock_mfdataset, bounds
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_mfdataset.return_value = _make_mock_dataset()

        from met_timeseries.sources.nldas import fetch_nldas

        result = fetch_nldas(bounds, 2010, 1, variables=["APCP", "TMP"])

        assert "APCP" in result
        assert "TMP" in result
        assert "DSWRF" not in result

    @patch("met_timeseries.sources.nldas.xr.open_mfdataset")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_dataset_has_time_lat_lon_dims(
        self, mock_login, mock_search, mock_open, mock_mfdataset, bounds
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_mfdataset.return_value = _make_mock_dataset()

        from met_timeseries.sources.nldas import fetch_nldas

        result = fetch_nldas(bounds, 2010, 1, variables=["APCP"])

        assert "time" in result.dims
        assert "lat" in result.dims
        assert "lon" in result.dims


class TestCachingMechanism:
    """Verify that the caching mechanism works correctly."""

    @patch("met_timeseries.sources.nldas.xr.open_mfdataset")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_writes_to_cache(
        self, mock_login, mock_search, mock_open, mock_mfdataset, bounds, tmp_path
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_mfdataset.return_value = _make_mock_dataset()

        from met_timeseries.sources.nldas import fetch_nldas

        fetch_nldas(bounds, 2010, 1, cache_dir=str(tmp_path))

        cache_file = tmp_path / "nldas" / "201001.nc"
        assert cache_file.exists()

    def test_loads_from_cache(self, bounds, tmp_path):
        # Pre-populate the cache
        cache_dir = tmp_path / "nldas"
        cache_dir.mkdir()
        ds = _make_mock_dataset()
        ds.to_netcdf(cache_dir / "201001.nc")

        with patch("earthaccess.login") as mock_login:
            from met_timeseries.sources.nldas import fetch_nldas

            result = fetch_nldas(bounds, 2010, 1, cache_dir=str(tmp_path))

        # earthaccess.login must NOT be called when loading from cache
        mock_login.assert_not_called()
        assert "APCP" in result
