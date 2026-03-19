"""
Tests for met_timeseries.sources.nldas — earthaccess-based implementation.
"""

from __future__ import annotations

import datetime
from unittest.mock import patch, MagicMock, call

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box


@pytest.fixture()
def bounds():
    from met_timeseries.sources.base import BoundingBox

    return BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)


def _make_mock_dataset(variables=("APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"), n_times=2):
    """Create a small mock xarray Dataset mimicking NLDAS-2 output."""
    times = pd.date_range("2010-01-01", periods=n_times, freq="h")
    lats = np.array([44.9375, 45.0625, 45.9375, 46.0625])
    lons = np.array([-110.0625, -109.9375, -109.0625, -108.9375])

    data_vars = {
        var: (["time", "lat", "lon"], np.ones((n_times, 4, 4), dtype=np.float32))
        for var in variables
    }

    return xr.Dataset(
        data_vars,
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _make_single_granule_dataset(t="2010-01-01T00:00:00", variables=None):
    """Create a tiny single-timestep Dataset for granule mocking."""
    if variables is None:
        variables = ("APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD")
    times = pd.DatetimeIndex([t])
    lats = np.array([44.9375, 45.0625, 45.9375, 46.0625])
    lons = np.array([-110.0625, -109.9375, -109.0625, -108.9375])
    data_vars = {
        var: (["time", "lat", "lon"], np.ones((1, 4, 4), dtype=np.float32))
        for var in variables
    }
    return xr.Dataset(data_vars, coords={"time": times, "lat": lats, "lon": lons})


class TestFetchNldasGridCallsEarthaccess:
    """Verify that fetch_nldas_grid calls earthaccess correctly."""

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_calls_login(self, mock_login, mock_search, mock_open, mock_subset, bounds):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.return_value = _make_single_granule_dataset()

        from met_timeseries.sources.nldas import fetch_nldas_grid

        fetch_nldas_grid(bounds, 2010, 1)

        mock_login.assert_called_once()

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_calls_search_data_with_correct_params(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.return_value = _make_single_granule_dataset()

        from met_timeseries.sources.nldas import fetch_nldas_grid

        fetch_nldas_grid(bounds, 2010, 1)

        mock_search.assert_called_once_with(
            short_name="NLDAS_FORA0125_H",
            version="2.0",
            temporal=("2010-01-01", "2010-01-31"),
            bounding_box=(-110.0, 45.0, -109.0, 46.0),
        )

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_calls_open_with_search_results(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        granules = [MagicMock()]
        mock_search.return_value = granules
        mock_open.return_value = [MagicMock()]
        mock_subset.return_value = _make_single_granule_dataset()

        from met_timeseries.sources.nldas import fetch_nldas_grid

        fetch_nldas_grid(bounds, 2010, 1)

        mock_open.assert_called_once_with(granules)

    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_raises_when_no_granules_found(self, mock_login, mock_search, bounds):
        mock_search.return_value = []

        from met_timeseries.sources.nldas import fetch_nldas_grid

        with pytest.raises(RuntimeError, match="No NLDAS-2 granules found"):
            fetch_nldas_grid(bounds, 2010, 1)

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_dataset_has_requested_variables(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        # Mock returns only the requested variables (as the real helper would)
        mock_subset.return_value = _make_single_granule_dataset(variables=("APCP", "TMP"))

        from met_timeseries.sources.nldas import fetch_nldas_grid

        result = fetch_nldas_grid(bounds, 2010, 1, variables=["APCP", "TMP"])

        assert "APCP" in result
        assert "TMP" in result
        assert "DSWRF" not in result

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_dataset_has_time_lat_lon_dims(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.return_value = _make_single_granule_dataset()

        from met_timeseries.sources.nldas import fetch_nldas_grid

        result = fetch_nldas_grid(bounds, 2010, 1, variables=["APCP"])

        assert "time" in result.dims
        assert "lat" in result.dims
        assert "lon" in result.dims


class TestCachingMechanism:
    """Verify that the caching mechanism works correctly."""

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_writes_to_cache(
        self, mock_login, mock_search, mock_open, mock_subset, bounds, tmp_path
    ):
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.return_value = _make_single_granule_dataset()

        from met_timeseries.sources.nldas import fetch_nldas_grid

        fetch_nldas_grid(bounds, 2010, 1, cache_dir=str(tmp_path))

        cache_file = tmp_path / "nldas" / "201001.nc"
        assert cache_file.exists()

    def test_loads_from_cache(self, bounds, tmp_path):
        # Pre-populate the cache
        cache_dir = tmp_path / "nldas"
        cache_dir.mkdir()
        ds = _make_mock_dataset()
        ds.to_netcdf(cache_dir / "201001.nc")

        with patch("earthaccess.login") as mock_login:
            from met_timeseries.sources.nldas import fetch_nldas_grid

            result = fetch_nldas_grid(bounds, 2010, 1, cache_dir=str(tmp_path))

        # earthaccess.login must NOT be called when loading from cache
        mock_login.assert_not_called()
        assert "APCP" in result


class TestMultiMonthYearRanges:
    """Tests for multi-month/year temporal range support."""

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_no_month_iterates_months(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        """When month=None, search_data is called once per month in the range."""
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.side_effect = [
            _make_single_granule_dataset(f"2010-{m:02d}-01T00:00:00")
            for m in range(1, 13)
        ]

        from met_timeseries.sources.nldas import fetch_nldas_grid

        fetch_nldas_grid(bounds, 2010, month=None, end_year=2010)

        assert mock_search.call_count == 12

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_multi_year(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        """year=2010, month=None, end_year=2011 should span 24 months."""
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]

        months_2010 = [_make_single_granule_dataset(f"2010-{m:02d}-01T00:00:00") for m in range(1, 13)]
        months_2011 = [_make_single_granule_dataset(f"2011-{m:02d}-01T00:00:00") for m in range(1, 13)]
        mock_subset.side_effect = months_2010 + months_2011

        from met_timeseries.sources.nldas import fetch_nldas_grid

        result = fetch_nldas_grid(bounds, 2010, month=None, end_year=2011)

        assert mock_search.call_count == 24
        assert len(result["time"]) == 24

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_end_year_defaults_to_current(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        """When end_year=None and month=None, range extends to the current year."""
        current_year = datetime.datetime.now().year
        expected_months = (current_year - 2010 + 1) * 12

        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.side_effect = [
            _make_single_granule_dataset(f"{y}-{m:02d}-01T00:00:00")
            for y in range(2010, current_year + 1)
            for m in range(1, 13)
        ]

        from met_timeseries.sources.nldas import fetch_nldas_grid

        fetch_nldas_grid(bounds, 2010, month=None, end_year=None)

        assert mock_search.call_count == expected_months

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_single_month_backward_compatible(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        """Calling with month=1 still results in a single search_data call."""
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.return_value = _make_single_granule_dataset()

        from met_timeseries.sources.nldas import fetch_nldas_grid

        result = fetch_nldas_grid(bounds, 2010, month=1)

        mock_search.assert_called_once()
        assert len(result["time"]) == 1

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_multi_month_uses_cache(
        self, mock_login, mock_search, mock_open, mock_subset, bounds, tmp_path
    ):
        """Cached months are loaded from disk; only uncached months trigger earthaccess."""
        # Pre-cache January 2010
        cache_dir = tmp_path / "nldas"
        cache_dir.mkdir()
        jan_ds = _make_single_granule_dataset("2010-01-01T00:00:00")
        jan_ds.to_netcdf(cache_dir / "201001.nc")

        # February is not cached
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        mock_subset.return_value = _make_single_granule_dataset("2010-02-01T00:00:00")

        from met_timeseries.sources.nldas import fetch_nldas_grid

        fetch_nldas_grid(bounds, 2010, month=None, end_year=2010,
                         cache_dir=str(tmp_path),
                         variables=["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"])

        # search_data should have been called for each non-cached month (months 2-12)
        assert mock_search.call_count == 11


class TestConcurrentDownloads:
    """Tests for concurrent granule download behaviour."""

    @patch("met_timeseries.sources.nldas.concurrent.futures.ThreadPoolExecutor")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_uses_thread_pool(
        self, mock_login, mock_search, mock_open, mock_executor_cls, bounds
    ):
        """ThreadPoolExecutor is used when fetching granules."""
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]

        # Set up a working executor mock that actually processes futures
        granule_ds = _make_single_granule_dataset()
        mock_future = MagicMock()
        mock_future.result.return_value = granule_ds
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        with patch("met_timeseries.sources.nldas.concurrent.futures.as_completed",
                   return_value=[mock_future]):
            from met_timeseries.sources.nldas import fetch_nldas_grid
            fetch_nldas_grid(bounds, 2010, 1)

        mock_executor_cls.assert_called_once()

    @patch("met_timeseries.sources.nldas.concurrent.futures.ThreadPoolExecutor")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_max_connections_parameter(
        self, mock_login, mock_search, mock_open, mock_executor_cls, bounds
    ):
        """max_connections=4 is passed as max_workers to ThreadPoolExecutor."""
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]

        granule_ds = _make_single_granule_dataset()
        mock_future = MagicMock()
        mock_future.result.return_value = granule_ds
        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        with patch("met_timeseries.sources.nldas.concurrent.futures.as_completed",
                   return_value=[mock_future]):
            from met_timeseries.sources.nldas import fetch_nldas_grid
            fetch_nldas_grid(bounds, 2010, 1, max_connections=4)

        mock_executor_cls.assert_called_once_with(max_workers=4)

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_granule_subset_before_concat(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        """Returned Dataset lat/lon dims only cover requested bounds, not full CONUS."""
        mock_search.return_value = [MagicMock()]
        mock_open.return_value = [MagicMock()]
        # Return dataset already spatially subsetted within the bounds
        subsetted = xr.Dataset(
            {"APCP": (["time", "lat", "lon"], np.ones((1, 2, 2), dtype=np.float32))},
            coords={
                "time": pd.DatetimeIndex(["2010-01-01"]),
                "lat": np.array([45.0625, 45.9375]),
                "lon": np.array([-109.9375, -109.0625]),
            },
        )
        mock_subset.return_value = subsetted

        from met_timeseries.sources.nldas import fetch_nldas_grid

        result = fetch_nldas_grid(bounds, 2010, 1, variables=["APCP"])

        # lat/lon should only span the bounds, not the full grid
        assert result["lat"].values.min() >= bounds.south
        assert result["lat"].values.max() <= bounds.north
        assert result["lon"].values.min() >= bounds.west
        assert result["lon"].values.max() <= bounds.east

    def test_open_and_subset_granule_helper(self, bounds):
        """_open_and_subset_granule subsets variables and spatial extent."""
        full_ds = xr.Dataset(
            {
                "APCP": (["time", "lat", "lon"], np.ones((1, 4, 4), dtype=np.float32)),
                "TMP": (["time", "lat", "lon"], np.ones((1, 4, 4), dtype=np.float32)),
                "EXTRA": (["time", "lat", "lon"], np.ones((1, 4, 4), dtype=np.float32)),
            },
            coords={
                "time": pd.DatetimeIndex(["2010-01-01"]),
                "lat": np.array([44.0, 45.0, 46.0, 47.0]),
                "lon": np.array([-111.0, -110.0, -109.0, -108.0]),
            },
        )

        # Mock the file object — xr.open_dataset is patched to return the full dataset
        mock_file = MagicMock()

        with patch("met_timeseries.sources.nldas.xr.open_dataset", return_value=full_ds):
            from met_timeseries.sources.nldas import _open_and_subset_granule

            result = _open_and_subset_granule(mock_file, ["APCP", "TMP"], bounds)

        assert "APCP" in result
        assert "TMP" in result
        assert "EXTRA" not in result
        assert result["lat"].values.min() >= bounds.south
        assert result["lat"].values.max() <= bounds.north

    @patch("met_timeseries.sources.nldas._open_and_subset_granule")
    @patch("earthaccess.open")
    @patch("earthaccess.search_data")
    @patch("earthaccess.login")
    def test_fetch_grid_skips_failed_granule(
        self, mock_login, mock_search, mock_open, mock_subset, bounds
    ):
        """If one granule raises, it is skipped and the rest of the month succeeds."""
        mock_search.return_value = [MagicMock(), MagicMock()]
        mock_open.return_value = [MagicMock(), MagicMock()]

        good_ds = _make_single_granule_dataset("2010-01-01T01:00:00")
        mock_subset.side_effect = [RuntimeError("broken granule"), good_ds]

        from met_timeseries.sources.nldas import fetch_nldas_grid

        result = fetch_nldas_grid(bounds, 2010, 1)

        # Should succeed and only contain the good granule's data
        assert "time" in result.dims
        assert len(result["time"]) == 1


class TestParseGiovanniResponse:
    """Verify the Giovanni CSV response parser."""

    _SAMPLE = (
        "param_short_name,Tair\n"
        "param_name,Near surface air temperature\n"
        "unit,K\n"
        "lat,35.0625\n"
        "lon,-83.9375\n"
        "start_time,2010-01-01T00:00:00\n"
        "end_time,2010-01-01T02:00:00\n"
        "temporal_resolution,hourly\n"
        "grid_type,regular\n"
        "grid_resolution,0.125\n"
        "source,NLDAS_FORA0125_H_2.0\n"
        "missing_value,-9999.0\n"
        "num_values,3\n"
        "Timestamp,Near surface air temperature\n"
        "2010-01-01T00:00:00,274.663\n"
        "2010-01-01T01:00:00,273.112\n"
        "2010-01-01T02:00:00,271.500\n"
    )

    def test_returns_series(self):
        from met_timeseries.sources.nldas import _parse_giovanni_response

        result = _parse_giovanni_response(self._SAMPLE)
        assert isinstance(result, pd.Series)

    def test_correct_values(self):
        from met_timeseries.sources.nldas import _parse_giovanni_response

        result = _parse_giovanni_response(self._SAMPLE)
        assert result.iloc[0] == pytest.approx(274.663)
        assert len(result) == 3

    def test_datetime_index(self):
        from met_timeseries.sources.nldas import _parse_giovanni_response

        result = _parse_giovanni_response(self._SAMPLE)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestFetchGiovanniCell:
    """Verify _fetch_giovanni_cell performs HTTP fetch and returns raw text."""

    def test_raises_on_unknown_variable(self):
        from met_timeseries.sources.nldas import _fetch_giovanni_cell

        with pytest.raises(ValueError, match="Unknown variable"):
            _fetch_giovanni_cell(
                35.0, -85.0, "NOTAVAR",
                "2010-01-01T00:00:00", "2010-01-31T23:00:00",
                _token="dummy",
            )

    @patch("requests.get")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_uses_bearer_token(
        self, mock_login, mock_get_token, mock_requests_get
    ):
        mock_get_token.return_value = {"access_token": "test-token"}
        mock_resp = MagicMock()
        mock_resp.text = "dummy"
        mock_requests_get.return_value = mock_resp

        from met_timeseries.sources.nldas import _fetch_giovanni_cell

        result = _fetch_giovanni_cell(
            35.0, -85.0, "APCP",
            "2010-01-01T00:00:00", "2010-01-31T23:00:00",
        )

        call_kwargs = mock_requests_get.call_args
        assert "Authorization" in call_kwargs.kwargs["headers"]
        assert call_kwargs.kwargs["headers"]["Authorization"].startswith("Bearer ")
        assert result == "dummy"

    @patch("requests.get")
    def test_uses_provided_token(self, mock_requests_get):
        mock_resp = MagicMock()
        mock_resp.text = "dummy"
        mock_requests_get.return_value = mock_resp

        from met_timeseries.sources.nldas import _fetch_giovanni_cell

        result = _fetch_giovanni_cell(
            35.0, -85.0, "APCP",
            "2010-01-01T00:00:00", "2010-01-31T23:00:00",
            _token="my-token",
        )

        call_kwargs = mock_requests_get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-token"
        assert result == "dummy"

    @patch("requests.get")
    def test_returns_raw_text(self, mock_requests_get):
        mock_resp = MagicMock()
        mock_resp.text = "raw response text"
        mock_requests_get.return_value = mock_resp

        from met_timeseries.sources.nldas import _fetch_giovanni_cell

        result = _fetch_giovanni_cell(
            35.0, -85.0, "APCP",
            "2010-01-01T00:00:00", "2010-01-31T23:00:00",
            _token="tok",
        )
        assert isinstance(result, str)
        assert result == "raw response text"


class TestCacheGiovanniResponse:
    """Verify _cache_giovanni_response caching of raw text."""

    _SAMPLE_TEXT = (
        "param_short_name,Tair\n"
        "param_name,Near surface air temperature\n"
        "unit,K\n"
        "lat,35.0625\n"
        "lon,-83.9375\n"
        "start_time,2010-01-01T00:00:00\n"
        "end_time,2010-01-01T02:00:00\n"
        "temporal_resolution,hourly\n"
        "grid_type,regular\n"
        "grid_resolution,0.125\n"
        "source,NLDAS_FORA0125_H_2.0\n"
        "missing_value,-9999.0\n"
        "num_values,3\n"
        "Timestamp,Near surface air temperature\n"
        "2010-01-01T00:00:00,274.663\n"
        "2010-01-01T01:00:00,273.112\n"
        "2010-01-01T02:00:00,271.500\n"
    )

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    def test_returns_text_from_cache(self, mock_fetch, tmp_path):
        from met_timeseries.sources.nldas import _cache_giovanni_response

        # Pre-populate cache (new key format: start_end dates)
        cache_file = tmp_path / "giovanni" / "35.0625_-83.9375_APCP_20100101_20100131.txt"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text(self._SAMPLE_TEXT)

        result = _cache_giovanni_response(
            35.0625, -83.9375, "APCP",
            "2010-01-01T00:00:00", "2010-01-31T23:00:00",
            cache_dir=str(tmp_path), _token="tok",
        )

        assert result == self._SAMPLE_TEXT
        mock_fetch.assert_not_called()

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    def test_fetches_and_caches_when_missing(self, mock_fetch, tmp_path):
        from met_timeseries.sources.nldas import _cache_giovanni_response

        mock_fetch.return_value = self._SAMPLE_TEXT

        result = _cache_giovanni_response(
            35.0625, -83.9375, "APCP",
            "2010-01-01T00:00:00", "2010-01-31T23:00:00",
            cache_dir=str(tmp_path), _token="tok",
        )

        assert result == self._SAMPLE_TEXT
        mock_fetch.assert_called_once()
        cache_file = tmp_path / "giovanni" / "35.0625_-83.9375_APCP_20100101_20100131.txt"
        assert cache_file.exists()
        assert cache_file.read_text() == self._SAMPLE_TEXT

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    def test_no_cache_dir_returns_text(self, mock_fetch):
        from met_timeseries.sources.nldas import _cache_giovanni_response

        mock_fetch.return_value = self._SAMPLE_TEXT

        result = _cache_giovanni_response(
            35.0625, -83.9375, "APCP",
            "2010-01-01T00:00:00", "2010-01-31T23:00:00",
            _token="tok",
        )

        assert result == self._SAMPLE_TEXT
        mock_fetch.assert_called_once()


class TestProcessNldas:
    """Verify process_nldas orchestrates download + weight derivation + averaging."""

    def _make_weights(self):
        return pd.DataFrame({
            "metzone_id": [1],
            "lat_center": [35.0625],
            "lon_center": [-83.9375],
            "weight": [1.0],
        })

    @patch("met_timeseries.sources.nldas.download_datarods")
    def test_returns_dict_of_dataframes(self, mock_download):
        from met_timeseries.sources.base import BoundingBox
        from met_timeseries.sources.nldas import process_nldas

        times = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_download.return_value = {
            (35.0625, -83.9375): {"APCP": pd.Series([1.0, 2.0, 3.0], index=times)}
        }

        bounds = BoundingBox(west=-84.0, east=-83.875, south=35.0, north=35.125)
        result = process_nldas(
            bounds, 2010, 1, variables=["APCP"], weights=self._make_weights(),
        )

        assert isinstance(result, dict)
        assert "APCP" in result
        assert isinstance(result["APCP"], pd.DataFrame)

    @patch("met_timeseries.sources.nldas.download_datarods")
    def test_calls_download_datarods_with_bounds(self, mock_download):
        from met_timeseries.sources.base import BoundingBox
        from met_timeseries.sources.nldas import process_nldas

        times = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_download.return_value = {
            (35.0625, -83.9375): {"APCP": pd.Series([1.0, 2.0, 3.0], index=times)}
        }

        bounds = BoundingBox(west=-84.0, east=-83.875, south=35.0, north=35.125)
        process_nldas(
            bounds, 2010, 1, variables=["APCP"], weights=self._make_weights(),
        )

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args.kwargs
        assert call_kwargs["bounds"] is bounds

    @patch("met_timeseries.sources.nldas.compute_nldas_weights")
    @patch("met_timeseries.sources.nldas.download_datarods")
    def test_auto_derives_weights_when_none(self, mock_download, mock_weights):
        """When weights=None, process_nldas should derive weights from the bounding box."""
        from met_timeseries.sources.base import BoundingBox
        from met_timeseries.sources.nldas import process_nldas

        times = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_download.return_value = {
            (35.0625, -83.9375): {"APCP": pd.Series([1.0, 2.0, 3.0], index=times)}
        }
        mock_weights.return_value = pd.DataFrame({
            "metzone_id": ["bbox"],
            "lat_center": [35.0625],
            "lon_center": [-83.9375],
            "weight": [1.0],
        })

        bounds = BoundingBox(west=-84.0, east=-83.875, south=35.0, north=35.125)
        result = process_nldas(bounds, 2010, 1, variables=["APCP"])
        assert "APCP" in result
        mock_weights.assert_called_once()

    @patch("met_timeseries.sources.nldas.download_datarods")
    def test_uses_provided_weights(self, mock_download):
        from met_timeseries.sources.base import BoundingBox
        from met_timeseries.sources.nldas import process_nldas

        times = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_download.return_value = {
            (35.0625, -83.9375): {"APCP": pd.Series([10.0, 20.0, 30.0], index=times)}
        }

        bounds = BoundingBox(west=-84.0, east=-83.875, south=35.0, north=35.125)
        weights = pd.DataFrame({
            "metzone_id": [1],
            "lat_center": [35.0625],
            "lon_center": [-83.9375],
            "weight": [1.0],
        })
        result = process_nldas(bounds, 2010, 1, variables=["APCP"], weights=weights)
        assert "APCP" in result
        # With weight=1.0 on the single cell, values should match raw data
        pd.testing.assert_series_equal(
            result["APCP"][1],
            pd.Series([10.0, 20.0, 30.0], index=times),
            check_names=False,
        )


class TestGenerateNldasGrid:
    """Verify generate_nldas_grid produces a correct GeoDataFrame."""

    def test_returns_geodataframe(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        result = generate_nldas_grid(west=-100.0, south=40.0, east=-99.0, north=41.0)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_crs_is_epsg4326(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        result = generate_nldas_grid(west=-100.0, south=40.0, east=-99.0, north=41.0)
        assert result.crs.to_epsg() == 4326

    def test_columns_present(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        result = generate_nldas_grid(west=-100.0, south=40.0, east=-99.0, north=41.0)
        assert "lat_center" in result.columns
        assert "lon_center" in result.columns
        assert "geometry" in result.columns

    def test_cell_count_for_small_extent(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        # 1° × 1° extent at 0.125° resolution => 9 × 9 = 81 cells
        result = generate_nldas_grid(
            west=-100.0, south=40.0, east=-99.0, north=41.0, resolution=0.125
        )
        assert len(result) == 81

    def test_cell_geometry_is_correct_size(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        result = generate_nldas_grid(west=-100.0, south=40.0, east=-99.0, north=41.0)
        cell = result.geometry.iloc[0]
        minx, miny, maxx, maxy = cell.bounds
        assert (maxx - minx) == pytest.approx(0.125, abs=1e-6)
        assert (maxy - miny) == pytest.approx(0.125, abs=1e-6)

    def test_centroid_matches_lat_lon_center(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        result = generate_nldas_grid(west=-100.0, south=40.0, east=-99.0, north=41.0)
        for _, row in result.iterrows():
            centroid = row["geometry"].centroid
            assert centroid.x == pytest.approx(row["lon_center"], abs=1e-4)
            assert centroid.y == pytest.approx(row["lat_center"], abs=1e-4)


class TestComputeNldasWeights:
    """Verify compute_nldas_weights produces correct weight tables."""

    def _make_polygon_gdf(self, geom, poly_id=1):
        return gpd.GeoDataFrame(
            {"metzone_id": [poly_id], "geometry": [geom]},
            crs="EPSG:4326",
        )

    def test_weights_sum_to_one(self):
        from met_timeseries.sources.nldas import compute_nldas_weights

        poly = box(-99.5, 40.3, -99.2, 40.6)
        gdf = self._make_polygon_gdf(poly)
        result = compute_nldas_weights(gdf)
        total = result.groupby("metzone_id")["weight"].sum()
        assert total.iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_output_columns(self):
        from met_timeseries.sources.nldas import compute_nldas_weights

        poly = box(-99.5, 40.3, -99.2, 40.6)
        gdf = self._make_polygon_gdf(poly)
        result = compute_nldas_weights(gdf)
        assert list(result.columns) == ["metzone_id", "lat_center", "lon_center", "weight"]

    def test_correct_cells_identified(self):
        from met_timeseries.sources.nldas import compute_nldas_weights, generate_nldas_grid

        # With west=-100.5, cell centers land at -100.5, -100.375, ..., -100.0, -99.875, ..., -99.5
        # Use cell centered at (-100.0, 40.0) which is on this grid
        cell_lon, cell_lat = -100.0, 40.0
        # Tiny polygon (0.02° × 0.02°) entirely within the 0.125° cell
        poly = box(cell_lon - 0.01, cell_lat - 0.01, cell_lon + 0.01, cell_lat + 0.01)
        gdf = self._make_polygon_gdf(poly)
        grid = generate_nldas_grid(west=-100.5, south=39.5, east=-99.5, north=40.5)
        result = compute_nldas_weights(gdf, nldas_grid=grid)
        assert len(result) == 1
        assert result.iloc[0]["weight"] == pytest.approx(1.0, abs=1e-6)

    def test_multiple_polygons_each_sum_to_one(self):
        from met_timeseries.sources.nldas import compute_nldas_weights

        poly1 = box(-99.5, 40.3, -99.2, 40.6)
        poly2 = box(-98.5, 41.3, -98.2, 41.6)
        gdf = gpd.GeoDataFrame(
            {"metzone_id": [1, 2], "geometry": [poly1, poly2]},
            crs="EPSG:4326",
        )
        result = compute_nldas_weights(gdf)
        totals = result.groupby("metzone_id")["weight"].sum()
        assert totals[1] == pytest.approx(1.0, abs=1e-6)
        assert totals[2] == pytest.approx(1.0, abs=1e-6)


class TestDownloadDatarods:
    """Verify download_datarods raw download behaviour."""

    def _make_bounds(self):
        from met_timeseries.sources.base import BoundingBox
        return BoundingBox(west=-84.0, south=35.0, east=-83.875, north=35.25)

    def _make_series(self):
        return pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range("2010-01-01", periods=3, freq="h"),
        )

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_returns_cell_data_dict(self, mock_login, mock_get_token, mock_cache, mock_parse):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"
        mock_parse.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        result = download_datarods(self._make_bounds(), 2010, 1, variables=["APCP"])

        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(val, dict)
            assert isinstance(val["APCP"], pd.Series)

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_authenticates_once(self, mock_login, mock_get_token, mock_cache, mock_parse):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"
        mock_parse.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        download_datarods(self._make_bounds(), 2010, 1, variables=["APCP", "TMP"])

        mock_login.assert_called_once()
        mock_get_token.assert_called_once()

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_fetches_all_variables_for_each_cell(
        self, mock_login, mock_get_token, mock_cache, mock_parse
    ):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"
        mock_parse.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        variables = ["APCP", "TMP"]
        result = download_datarods(self._make_bounds(), 2010, 1, variables=variables)

        n_cells = len(result)
        # Each cell should have all requested variables
        assert mock_cache.call_count == n_cells * len(variables)

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_passes_cache_dir(self, mock_login, mock_get_token, mock_cache, mock_parse):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"
        mock_parse.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        download_datarods(
            self._make_bounds(), 2010, 1, variables=["APCP"], cache_dir="/tmp/cache"
        )

        for call in mock_cache.call_args_list:
            assert call.kwargs["cache_dir"] == "/tmp/cache"

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_no_month_fetches_yearly_chunks(self, mock_login, mock_get_token, mock_cache, mock_parse):
        """When month=None, download_datarods should make one request per year (not per month)."""
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"
        mock_parse.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        result = download_datarods(
            self._make_bounds(), year=2010, month=None, variables=["APCP"], end_year=2010,
        )

        n_cells = len(result)
        # 1 year × n_cells × 1 variable (yearly chunking, not monthly)
        assert mock_cache.call_count == n_cells * 1

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_no_month_multi_year(self, mock_login, mock_get_token, mock_cache, mock_parse):
        """No month with year range 2010-2011 should fetch 2 yearly chunks."""
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"
        mock_parse.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        result = download_datarods(
            self._make_bounds(), year=2010, month=None, variables=["APCP"], end_year=2011,
        )

        n_cells = len(result)
        # 2 years × n_cells × 1 variable
        assert mock_cache.call_count == n_cells * 2

    @patch("met_timeseries.sources.nldas.get_nldas_gridcells")
    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_no_month_concatenates_yearly_series(
        self, mock_login, mock_get_token, mock_cache, mock_parse, mock_gridcells
    ):
        """When multiple years are fetched, the series should be concatenated."""
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"

        # Mock a single grid cell
        mock_gridcells.return_value = gpd.GeoDataFrame(
            {"lat_center": [35.0625], "lon_center": [-83.9375],
             "geometry": [box(-84.0, 35.0, -83.875, 35.125)]},
            crs="EPSG:4326",
        )

        # Mock 2 yearly series (1 cell × 1 variable × 2 years)
        yearly_series = [
            pd.Series([1.0, 2.0], index=pd.date_range("2010-01-01", periods=2, freq="h")),
            pd.Series([3.0, 4.0], index=pd.date_range("2011-01-01", periods=2, freq="h")),
        ]
        mock_parse.side_effect = yearly_series

        from met_timeseries.sources.nldas import download_datarods

        result = download_datarods(
            self._make_bounds(), year=2010, month=None, variables=["APCP"], end_year=2011,
        )

        cell_key = (35.0625, -83.9375)
        series = result[cell_key]["APCP"]
        # Should have 4 data points (2 per year × 2 years)
        assert len(series) == 4

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("met_timeseries.sources.nldas._cache_giovanni_response")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_default_year_is_1995(self, mock_login, mock_get_token, mock_cache, mock_parse):
        """When no year is provided, default should be 1995."""
        mock_get_token.return_value = {"access_token": "tok"}
        mock_cache.return_value = "raw text"
        mock_parse.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        # Call with month specified to keep the test simple (single month)
        download_datarods(self._make_bounds(), month=1, variables=["APCP"])

        # The cache calls should use start_date beginning with 1995
        for call in mock_cache.call_args_list:
            assert call.kwargs["start_date"].startswith("1995-")


class TestComputeWeightedAverages:
    """Verify compute_weighted_averages spatial aggregation."""

    def _make_cell_data(self, cells, variables, values=None):
        times = pd.date_range("2010-01-01", periods=3, freq="h")
        cell_data = {}
        for i, cell in enumerate(cells):
            cell_data[cell] = {}
            for j, var in enumerate(variables):
                v = values[i][j] if values else float(i + j + 1)
                cell_data[cell][var] = pd.Series([v] * 3, index=times)
        return cell_data

    def test_returns_dict_of_dataframes(self):
        from met_timeseries.sources.nldas import compute_weighted_averages

        cell_data = self._make_cell_data([(35.0625, -83.9375)], ["APCP"])
        weights = pd.DataFrame(
            {
                "metzone_id": [1],
                "lat_center": [35.0625],
                "lon_center": [-83.9375],
                "weight": [1.0],
            }
        )
        result = compute_weighted_averages(cell_data, weights, variables=["APCP"])
        assert isinstance(result, dict)
        assert isinstance(result["APCP"], pd.DataFrame)

    def test_single_cell_weight_one(self):
        from met_timeseries.sources.nldas import compute_weighted_averages

        times = pd.date_range("2010-01-01", periods=3, freq="h")
        raw = pd.Series([10.0, 20.0, 30.0], index=times)
        cell_data = {(35.0625, -83.9375): {"APCP": raw}}
        weights = pd.DataFrame(
            {
                "metzone_id": [1],
                "lat_center": [35.0625],
                "lon_center": [-83.9375],
                "weight": [1.0],
            }
        )
        result = compute_weighted_averages(cell_data, weights, variables=["APCP"])
        pd.testing.assert_series_equal(
            result["APCP"][1], raw, check_names=False
        )

    def test_two_cells_equal_weight(self):
        from met_timeseries.sources.nldas import compute_weighted_averages

        times = pd.date_range("2010-01-01", periods=3, freq="h")
        s1 = pd.Series([10.0, 20.0, 30.0], index=times)
        s2 = pd.Series([20.0, 40.0, 60.0], index=times)
        cell_data = {
            (35.0625, -83.9375): {"APCP": s1},
            (35.1875, -83.9375): {"APCP": s2},
        }
        weights = pd.DataFrame(
            {
                "metzone_id": [1, 1],
                "lat_center": [35.0625, 35.1875],
                "lon_center": [-83.9375, -83.9375],
                "weight": [0.5, 0.5],
            }
        )
        result = compute_weighted_averages(cell_data, weights, variables=["APCP"])
        expected = (s1 + s2) / 2
        pd.testing.assert_series_equal(
            result["APCP"][1], expected, check_names=False
        )

    def test_infers_variables_from_cell_data(self):
        from met_timeseries.sources.nldas import compute_weighted_averages

        times = pd.date_range("2010-01-01", periods=2, freq="h")
        cell_data = {
            (35.0625, -83.9375): {
                "APCP": pd.Series([1.0, 2.0], index=times),
                "TMP": pd.Series([300.0, 301.0], index=times),
            }
        }
        weights = pd.DataFrame(
            {
                "metzone_id": [1],
                "lat_center": [35.0625],
                "lon_center": [-83.9375],
                "weight": [1.0],
            }
        )
        result = compute_weighted_averages(cell_data, weights)
        assert "APCP" in result
        assert "TMP" in result

    def test_multiple_polygons(self):
        from met_timeseries.sources.nldas import compute_weighted_averages

        times = pd.date_range("2010-01-01", periods=2, freq="h")
        cell_data = {
            (35.0625, -83.9375): {"APCP": pd.Series([1.0, 2.0], index=times)},
            (36.0625, -83.9375): {"APCP": pd.Series([3.0, 4.0], index=times)},
        }
        weights = pd.DataFrame(
            {
                "metzone_id": [1, 2],
                "lat_center": [35.0625, 36.0625],
                "lon_center": [-83.9375, -83.9375],
                "weight": [1.0, 1.0],
            }
        )
        result = compute_weighted_averages(cell_data, weights, variables=["APCP"])
        assert 1 in result["APCP"].columns
        assert 2 in result["APCP"].columns
