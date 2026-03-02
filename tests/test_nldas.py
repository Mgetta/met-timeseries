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


# ---------------------------------------------------------------------------
# Tests for generate_nldas_grid
# ---------------------------------------------------------------------------

class TestGenerateNldasGrid:
    """Tests for generate_nldas_grid()."""

    def test_returns_geodataframe(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        gdf = generate_nldas_grid(west=-84.0, south=35.0, east=-83.0, north=36.0)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_crs_is_epsg4326(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        gdf = generate_nldas_grid(west=-84.0, south=35.0, east=-83.0, north=36.0)
        assert gdf.crs.to_epsg() == 4326

    def test_columns_present(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        gdf = generate_nldas_grid(west=-84.0, south=35.0, east=-83.0, north=36.0)
        assert "lat_center" in gdf.columns
        assert "lon_center" in gdf.columns
        assert "geometry" in gdf.columns

    def test_cell_count_for_small_extent(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        # 1 degree extent / 0.125° resolution = 9 cells per axis (0.125 steps: 9 values)
        gdf = generate_nldas_grid(west=-84.0, south=35.0, east=-83.0, north=36.0, resolution=0.125)
        # lons: -84.0, -83.875, ..., -83.0  => 9 values
        # lats: 35.0, 35.125, ..., 36.0     => 9 values
        assert len(gdf) == 9 * 9

    def test_cell_geometry_is_correct_size(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        gdf = generate_nldas_grid(west=-84.0, south=35.0, east=-83.0, north=36.0)
        first_cell = gdf.iloc[0].geometry
        minx, miny, maxx, maxy = first_cell.bounds
        assert abs((maxx - minx) - 0.125) < 1e-8
        assert abs((maxy - miny) - 0.125) < 1e-8

    def test_centroid_matches_lat_lon_center(self):
        from met_timeseries.sources.nldas import generate_nldas_grid

        gdf = generate_nldas_grid(west=-84.0, south=35.0, east=-83.0, north=36.0)
        for _, row in gdf.iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            assert abs(cx - row["lon_center"]) < 1e-6
            assert abs(cy - row["lat_center"]) < 1e-6


# ---------------------------------------------------------------------------
# Tests for compute_nldas_weights
# ---------------------------------------------------------------------------

class TestComputeNldasWeights:
    """Tests for compute_nldas_weights()."""

    def _make_simple_polygon(self, metzone_id="zone1"):
        """Create a 0.25° × 0.25° square polygon exactly covering 4 NLDAS cells."""
        geom = box(-84.0625, 35.0625, -83.8125, 35.3125)
        return gpd.GeoDataFrame(
            {"metzone_id": [metzone_id], "geometry": [geom]}, crs="EPSG:4326"
        )

    def test_weights_sum_to_one(self):
        from met_timeseries.sources.nldas import compute_nldas_weights

        polygons = self._make_simple_polygon()
        weights = compute_nldas_weights(polygons)
        total = weights.groupby("metzone_id")["weight"].sum()
        assert abs(total.iloc[0] - 1.0) < 1e-6

    def test_output_columns(self):
        from met_timeseries.sources.nldas import compute_nldas_weights

        polygons = self._make_simple_polygon()
        weights = compute_nldas_weights(polygons)
        assert set(weights.columns) == {"metzone_id", "lat_center", "lon_center", "weight"}

    def test_correct_cells_identified(self):
        from met_timeseries.sources.nldas import compute_nldas_weights, generate_nldas_grid

        # Use the actual NLDAS grid so alignment is guaranteed
        grid = generate_nldas_grid()
        # Pick a cell and create a polygon well inside it
        cell = grid.iloc[0]
        cx, cy = cell.lon_center, cell.lat_center
        # Slightly inset so the polygon doesn't touch the cell boundary
        inner = box(cx - 0.02, cy - 0.02, cx + 0.02, cy + 0.02)
        polygons = gpd.GeoDataFrame(
            {"metzone_id": ["single"], "geometry": [inner]}, crs="EPSG:4326"
        )
        weights = compute_nldas_weights(polygons, nldas_grid=grid)
        # Should have exactly one cell with weight ≈ 1.0
        assert len(weights) == 1
        assert abs(weights.iloc[0]["weight"] - 1.0) < 1e-6

    def test_multiple_polygons_each_sum_to_one(self):
        from met_timeseries.sources.nldas import compute_nldas_weights

        geom1 = box(-84.0625, 35.0625, -83.8125, 35.3125)
        geom2 = box(-83.8125, 35.0625, -83.5625, 35.3125)
        polygons = gpd.GeoDataFrame(
            {
                "metzone_id": ["a", "b"],
                "geometry": [geom1, geom2],
            },
            crs="EPSG:4326",
        )
        weights = compute_nldas_weights(polygons)
        totals = weights.groupby("metzone_id")["weight"].sum()
        for t in totals:
            assert abs(t - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests for _parse_datarods_response
# ---------------------------------------------------------------------------

class TestParseDatarodsResponse:
    """Tests for _parse_datarods_response()."""

    _SAMPLE = (
        "Date&Time\tData\n"
        "2010-01-01T00\t0.0000\n"
        "2010-01-01T01\t1.2345\n"
        "2010-01-01T02\t2.5000\n"
    )

    def test_returns_series(self):
        from met_timeseries.sources.nldas import _parse_datarods_response

        result = _parse_datarods_response(self._SAMPLE)
        assert isinstance(result, pd.Series)

    def test_datetime_index(self):
        from met_timeseries.sources.nldas import _parse_datarods_response

        result = _parse_datarods_response(self._SAMPLE)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_correct_timestamps(self):
        from met_timeseries.sources.nldas import _parse_datarods_response

        result = _parse_datarods_response(self._SAMPLE)
        expected = pd.to_datetime(["2010-01-01T00", "2010-01-01T01", "2010-01-01T02"])
        assert list(result.index) == list(expected)

    def test_correct_values(self):
        from met_timeseries.sources.nldas import _parse_datarods_response

        result = _parse_datarods_response(self._SAMPLE)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.2345)
        assert result.iloc[2] == pytest.approx(2.5)

    def test_skips_comment_lines(self):
        from met_timeseries.sources.nldas import _parse_datarods_response

        text = (
            "# This is a comment\n"
            "Date&Time\tData\n"
            "2010-06-01T00\t5.0\n"
        )
        result = _parse_datarods_response(text)
        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests for fetch_nldas_datarods
# ---------------------------------------------------------------------------

class TestFetchNldasDatarods:
    """Tests for fetch_nldas_datarods() with mocked HTTP responses."""

    def _make_weights(self, polygon_id="zone1"):
        """Two cells with equal weights."""
        return pd.DataFrame(
            {
                "metzone_id": [polygon_id, polygon_id],
                "lat_center": [35.0625, 35.1875],
                "lon_center": [-83.9375, -83.9375],
                "weight": [0.5, 0.5],
            }
        )

    def _make_response_text(self, value: float, n_hours: int = 3) -> str:
        lines = ["Date&Time\tData"]
        base = pd.Timestamp("2010-01-01")
        for h in range(n_hours):
            ts = (base + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H")
            lines.append(f"{ts}\t{value:.4f}")
        return "\n".join(lines) + "\n"

    @patch("met_timeseries.sources.nldas._fetch_datarods_cell")
    def test_returns_dict_of_dataframes(self, mock_fetch):
        from met_timeseries.sources.nldas import fetch_nldas_datarods

        ts = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_fetch.return_value = pd.Series([1.0, 2.0, 3.0], index=ts)

        weights = self._make_weights()
        result = fetch_nldas_datarods(weights, 2010, 1, variables=["APCP"])

        assert isinstance(result, dict)
        assert "zone1" in result
        assert isinstance(result["zone1"], pd.DataFrame)

    @patch("met_timeseries.sources.nldas._fetch_datarods_cell")
    def test_weighted_average_is_correct(self, mock_fetch):
        from met_timeseries.sources.nldas import fetch_nldas_datarods

        ts = pd.date_range("2010-01-01", periods=3, freq="h")

        def side_effect(lat, lon, variable, year, month, cache_dir=None):
            if lat == 35.0625:
                return pd.Series([2.0, 4.0, 6.0], index=ts)
            else:
                return pd.Series([4.0, 8.0, 12.0], index=ts)

        mock_fetch.side_effect = side_effect

        weights = self._make_weights()
        result = fetch_nldas_datarods(weights, 2010, 1, variables=["APCP"])

        # Expected: 0.5 * [2,4,6] + 0.5 * [4,8,12] = [3,6,9]  then / 1.0
        expected = [3.0, 6.0, 9.0]
        actual = result["zone1"]["APCP"].tolist()
        for a, e in zip(actual, expected):
            assert a == pytest.approx(e)

    @patch("met_timeseries.sources.nldas._fetch_datarods_cell")
    def test_variable_columns_present(self, mock_fetch):
        from met_timeseries.sources.nldas import fetch_nldas_datarods

        ts = pd.date_range("2010-01-01", periods=3, freq="h")
        mock_fetch.return_value = pd.Series([1.0, 1.0, 1.0], index=ts)

        weights = self._make_weights()
        result = fetch_nldas_datarods(weights, 2010, 1, variables=["APCP", "TMP"])

        assert "APCP" in result["zone1"].columns
        assert "TMP" in result["zone1"].columns


# ---------------------------------------------------------------------------
# Tests for fetch_nldas strategy parameter
# ---------------------------------------------------------------------------

class TestFetchNldasStrategyParameter:
    """Tests for the strategy parameter in fetch_nldas."""

    def test_unknown_strategy_raises_value_error(self):
        from met_timeseries.sources.nldas import fetch_nldas
        from met_timeseries.sources.base import BoundingBox

        bounds = BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)
        with pytest.raises(ValueError, match="Unknown strategy"):
            fetch_nldas(bounds, 2010, 1, strategy="invalid")

    def test_datarods_without_weights_raises_value_error(self):
        from met_timeseries.sources.nldas import fetch_nldas
        from met_timeseries.sources.base import BoundingBox

        bounds = BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)
        with pytest.raises(ValueError, match="weights must be provided"):
            fetch_nldas(bounds, 2010, 1, strategy="datarods")

    @patch("met_timeseries.sources.nldas.fetch_nldas_datarods")
    def test_datarods_strategy_delegates_to_fetch_nldas_datarods(self, mock_datarods):
        from met_timeseries.sources.nldas import fetch_nldas
        from met_timeseries.sources.base import BoundingBox

        mock_datarods.return_value = {"zone1": pd.DataFrame()}
        bounds = BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)
        weights = pd.DataFrame(
            {
                "metzone_id": ["zone1"],
                "lat_center": [45.0625],
                "lon_center": [-109.9375],
                "weight": [1.0],
            }
        )

        result = fetch_nldas(bounds, 2010, 1, strategy="datarods", weights=weights)

        mock_datarods.assert_called_once()
        assert isinstance(result, dict)
