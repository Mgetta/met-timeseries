"""
Tests for met_timeseries.sources.nldas — earthaccess-based implementation.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

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
    """Verify _fetch_giovanni_cell raises on unknown variable."""

    def test_raises_on_unknown_variable(self):
        from met_timeseries.sources.nldas import _fetch_giovanni_cell

        with pytest.raises(ValueError, match="Unknown variable"):
            _fetch_giovanni_cell(35.0, -85.0, "NOTAVAR", 2010, 1, _token="dummy")

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("requests.get")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_uses_bearer_token(
        self, mock_login, mock_get_token, mock_requests_get, mock_parse
    ):
        mock_get_token.return_value = {"access_token": "test-token"}
        mock_resp = MagicMock()
        mock_resp.text = "dummy"
        mock_requests_get.return_value = mock_resp
        mock_parse.return_value = pd.Series(
            [1.0], index=pd.DatetimeIndex(["2010-01-01"])
        )

        from met_timeseries.sources.nldas import _fetch_giovanni_cell

        _fetch_giovanni_cell(35.0, -85.0, "APCP", 2010, 1)

        call_kwargs = mock_requests_get.call_args
        assert "Authorization" in call_kwargs.kwargs["headers"]
        assert call_kwargs.kwargs["headers"]["Authorization"].startswith("Bearer ")

    @patch("met_timeseries.sources.nldas._parse_giovanni_response")
    @patch("requests.get")
    def test_uses_provided_token(self, mock_requests_get, mock_parse):
        mock_resp = MagicMock()
        mock_resp.text = "dummy"
        mock_requests_get.return_value = mock_resp
        mock_parse.return_value = pd.Series(
            [1.0], index=pd.DatetimeIndex(["2010-01-01"])
        )

        from met_timeseries.sources.nldas import _fetch_giovanni_cell

        _fetch_giovanni_cell(35.0, -85.0, "APCP", 2010, 1, _token="my-token")

        call_kwargs = mock_requests_get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-token"


class TestFetchNldasDatarods:
    """Verify fetch_nldas_datarods orchestrates correctly."""

    def _make_weights(self):
        return pd.DataFrame(
            {
                "metzone_id": [1, 1],
                "lat_center": [35.0625, 35.0625],
                "lon_center": [-83.9375, -83.9375],
                "weight": [0.6, 0.4],
            }
        )

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_returns_dict_of_dataframes(
        self, mock_login, mock_get_token, mock_fetch_cell
    ):
        mock_get_token.return_value = {"access_token": "tok"}
        times = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_fetch_cell.return_value = pd.Series([1.0, 2.0, 3.0], index=times)

        from met_timeseries.sources.nldas import fetch_nldas_datarods

        result = fetch_nldas_datarods(
            self._make_weights(), 2010, 1, variables=["APCP"]
        )

        assert isinstance(result, dict)
        assert "APCP" in result
        assert isinstance(result["APCP"], pd.DataFrame)

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_authenticates_once(self, mock_login, mock_get_token, mock_fetch_cell):
        mock_get_token.return_value = {"access_token": "tok"}
        times = pd.date_range("2010-01-01", periods=2, freq="h")
        mock_fetch_cell.return_value = pd.Series([1.0, 2.0], index=times)

        from met_timeseries.sources.nldas import fetch_nldas_datarods

        fetch_nldas_datarods(
            self._make_weights(), 2010, 1, variables=["APCP", "TMP"]
        )

        mock_login.assert_called_once()
        mock_get_token.assert_called_once()


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


class TestFetchNldasStrategyParameter:
    """Verify the strategy parameter on fetch_nldas."""

    def test_unknown_strategy_raises_value_error(self):
        from met_timeseries.sources.base import BoundingBox
        from met_timeseries.sources.nldas import fetch_nldas

        bounds = BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)
        with pytest.raises(ValueError, match="Unknown strategy"):
            fetch_nldas(bounds, 2010, 1, strategy="invalid")

    def test_datarods_without_weights_raises_value_error(self):
        from met_timeseries.sources.base import BoundingBox
        from met_timeseries.sources.nldas import fetch_nldas

        bounds = BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)
        with pytest.raises(ValueError, match="weights must be provided"):
            fetch_nldas(bounds, 2010, 1, strategy="datarods")

    @patch("met_timeseries.sources.nldas.fetch_nldas_datarods")
    def test_datarods_strategy_delegates_to_fetch_nldas_datarods(self, mock_datarods):
        from met_timeseries.sources.base import BoundingBox
        from met_timeseries.sources.nldas import fetch_nldas

        mock_datarods.return_value = {"APCP": pd.DataFrame()}
        bounds = BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)
        weights = pd.DataFrame(
            {
                "metzone_id": [1],
                "lat_center": [45.0625],
                "lon_center": [-109.9375],
                "weight": [1.0],
            }
        )
        result = fetch_nldas(bounds, 2010, 1, strategy="datarods", weights=weights)
        mock_datarods.assert_called_once()
        assert result is mock_datarods.return_value


class TestDownloadNldas:
    """Verify download_nldas dispatching behaviour."""

    def _make_weights(self):
        return pd.DataFrame(
            {
                "metzone_id": [1],
                "lat_center": [35.0625],
                "lon_center": [-83.9375],
                "weight": [1.0],
            }
        )

    def test_unknown_method_raises_value_error(self):
        from met_timeseries.sources.nldas import download_nldas

        with pytest.raises(ValueError, match="Unknown download method"):
            download_nldas(self._make_weights(), 2010, 1, method="invalid")

    @patch("met_timeseries.sources.nldas.download_datarods")
    def test_dispatches_to_download_datarods(self, mock_download_datarods):
        from met_timeseries.sources.nldas import download_nldas

        mock_download_datarods.return_value = {}
        result = download_nldas(self._make_weights(), 2010, 1, method="datarods")
        mock_download_datarods.assert_called_once()
        assert result is mock_download_datarods.return_value

    @patch("met_timeseries.sources.nldas.download_datarods")
    def test_default_variables(self, mock_download_datarods):
        from met_timeseries.sources.nldas import download_nldas

        mock_download_datarods.return_value = {}
        download_nldas(self._make_weights(), 2010, 1)
        _, kwargs = mock_download_datarods.call_args
        assert kwargs["variables"] == ["APCP", "TMP", "DSWRF", "PEVAP", "UGRD", "VGRD"]


class TestDownloadDatarods:
    """Verify download_datarods raw download behaviour."""

    def _make_weights(self):
        return pd.DataFrame(
            {
                "metzone_id": [1, 1],
                "lat_center": [35.0625, 35.1875],
                "lon_center": [-83.9375, -83.9375],
                "weight": [0.6, 0.4],
            }
        )

    def _make_series(self):
        return pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range("2010-01-01", periods=3, freq="h"),
        )

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_returns_cell_data_dict(self, mock_login, mock_get_token, mock_fetch_cell):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_fetch_cell.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        result = download_datarods(self._make_weights(), 2010, 1, variables=["APCP"])

        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(val, dict)
            assert isinstance(val["APCP"], pd.Series)

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_authenticates_once(self, mock_login, mock_get_token, mock_fetch_cell):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_fetch_cell.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        download_datarods(self._make_weights(), 2010, 1, variables=["APCP", "TMP"])

        mock_login.assert_called_once()
        mock_get_token.assert_called_once()

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_fetches_all_cells_and_variables(
        self, mock_login, mock_get_token, mock_fetch_cell
    ):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_fetch_cell.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        variables = ["APCP", "TMP"]
        download_datarods(self._make_weights(), 2010, 1, variables=variables)

        # 2 unique cells × 2 variables = 4 calls
        assert mock_fetch_cell.call_count == 4

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_passes_cache_dir(self, mock_login, mock_get_token, mock_fetch_cell):
        mock_get_token.return_value = {"access_token": "tok"}
        mock_fetch_cell.return_value = self._make_series()

        from met_timeseries.sources.nldas import download_datarods

        download_datarods(
            self._make_weights(), 2010, 1, variables=["APCP"], cache_dir="/tmp/cache"
        )

        for call in mock_fetch_cell.call_args_list:
            assert call.kwargs["cache_dir"] == "/tmp/cache"


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


class TestFetchNldasDatarodsWrapper:
    """Verify fetch_nldas_datarods is a thin wrapper over download_datarods + compute_weighted_averages."""

    def _make_weights(self):
        return pd.DataFrame(
            {
                "metzone_id": [1],
                "lat_center": [35.0625],
                "lon_center": [-83.9375],
                "weight": [1.0],
            }
        )

    @patch("met_timeseries.sources.nldas.compute_weighted_averages")
    @patch("met_timeseries.sources.nldas.download_datarods")
    def test_calls_download_then_compute(
        self, mock_download, mock_compute
    ):
        from met_timeseries.sources.nldas import fetch_nldas_datarods

        times = pd.date_range("2010-01-01", periods=3, freq="h")
        fake_cell_data = {
            (35.0625, -83.9375): {"APCP": pd.Series([1.0, 2.0, 3.0], index=times)}
        }
        fake_result = {"APCP": pd.DataFrame({1: [1.0, 2.0, 3.0]}, index=times)}
        mock_download.return_value = fake_cell_data
        mock_compute.return_value = fake_result

        weights = self._make_weights()
        result = fetch_nldas_datarods(weights, 2010, 1, variables=["APCP"])

        mock_download.assert_called_once()
        mock_compute.assert_called_once()
        assert result is fake_result

    @patch("met_timeseries.sources.nldas._fetch_giovanni_cell")
    @patch("earthaccess.get_edl_token")
    @patch("earthaccess.login")
    def test_backward_compatible_return_type(
        self, mock_login, mock_get_token, mock_fetch_cell
    ):
        from met_timeseries.sources.nldas import fetch_nldas_datarods

        mock_get_token.return_value = {"access_token": "tok"}
        times = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_fetch_cell.return_value = pd.Series([1.0, 2.0, 3.0], index=times)

        result = fetch_nldas_datarods(
            self._make_weights(), 2010, 1, variables=["APCP"]
        )

        assert isinstance(result, dict)
        assert "APCP" in result
        assert isinstance(result["APCP"], pd.DataFrame)
