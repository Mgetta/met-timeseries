"""
Tests for met_timeseries.sources.prism — daily PRISM implementation.

All HTTP downloads are mocked so the tests run without network access.
"""

from __future__ import annotations

import datetime
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bounds():
    from met_timeseries.sources.base import BoundingBox

    return BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)


def _make_fake_dataarray(
    lats=None,
    lons=None,
    value: float = 1.0,
    name: str = "ppt",
    with_band: bool = True,
    use_yx: bool = True,
) -> xr.DataArray:
    """Return a tiny DataArray that simulates rasterio engine output.

    By default includes a ``band`` dimension and ``y``/``x`` coordinate names
    to mimic how rioxarray opens a TIF file.
    """
    if lats is None:
        lats = np.array([45.5, 45.0])
    if lons is None:
        lons = np.array([-110.0, -109.5, -109.0])

    lat_name = "y" if use_yx else "lat"
    lon_name = "x" if use_yx else "lon"

    data = np.full((len(lats), len(lons)), value, dtype=np.float32)

    if with_band:
        data = data[np.newaxis, ...]  # add band dim
        da = xr.DataArray(
            data,
            dims=["band", lat_name, lon_name],
            coords={"band": [1], lat_name: lats, lon_name: lons},
            name=name,
        )
    else:
        da = xr.DataArray(
            data,
            dims=[lat_name, lon_name],
            coords={lat_name: lats, lon_name: lons},
            name=name,
        )

    # Attach a minimal CRS-like attribute so rio methods don't crash
    da.attrs["crs"] = "EPSG:4326"
    return da


def _make_mock_fetch_single_day(value: float = 1.0):
    """Return a side_effect callable that produces DataArrays with lat/lon dims."""

    def _side_effect(bounds, date, variable, resolution="4km", cache_dir=None):
        lats = np.array([45.5, 45.0])
        lons = np.array([-110.0, -109.5, -109.0])
        data = np.full((len(lats), len(lons)), value, dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            name=variable,
        )
        return da

    return _side_effect


# ---------------------------------------------------------------------------
# Tests for fetch_prism
# ---------------------------------------------------------------------------

class TestFetchPrismInterface:
    """Test the public fetch_prism function."""

    def test_returns_dataset(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-06-15")

        assert isinstance(ds, xr.Dataset)

    def test_single_day_has_one_time_step(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-06-15")

        assert "time" in ds.dims
        assert ds.sizes["time"] == 1

    def test_multi_day_range(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-06-01", end="2020-06-05")

        assert ds.sizes["time"] == 5

    def test_end_same_as_start_gives_one_day(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-01-10", end="2020-01-10")

        assert ds.sizes["time"] == 1

    def test_has_lat_lon_dims(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-06-01", end="2020-06-03")

        assert "lat" in ds.dims
        assert "lon" in ds.dims

    def test_time_coords_are_datetime64(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-03-01", end="2020-03-03")

        times = ds.coords["time"].values
        assert np.issubdtype(times.dtype, np.datetime64)

    def test_time_coords_match_requested_dates(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-03-01", end="2020-03-03")

        times = ds.coords["time"].values.astype("datetime64[D]").astype(str).tolist()
        assert times == ["2020-03-01", "2020-03-02", "2020-03-03"]

    def test_default_variables(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-06-15")

        assert set(ds.data_vars) == {"ppt", "tmax", "tmin"}

    def test_custom_variables(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(
                bounds, start="2020-06-15", variables=["ppt", "tmean"]
            )

        assert set(ds.data_vars) == {"ppt", "tmean"}

    def test_raises_value_error_when_end_before_start(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with pytest.raises(ValueError, match="end.*before start"):
            fetch_prism(bounds, start="2020-06-15", end="2020-06-10")

    def test_calls_fetch_single_day_for_each_day_and_variable(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        mock_fn = MagicMock(side_effect=_make_mock_fetch_single_day())

        with patch("met_timeseries.sources.prism._fetch_single_day", mock_fn), \
             patch("met_timeseries.sources.prism.time.sleep"):
            fetch_prism(
                bounds, start="2020-01-01", end="2020-01-03", variables=["ppt", "tmax"]
            )

        # 3 days × 2 variables = 6 calls
        assert mock_fn.call_count == 6


# ---------------------------------------------------------------------------
# Tests for fetch_prism cache_dir parameter
# ---------------------------------------------------------------------------

class TestFetchPrismCacheDir:
    """Test that cache_dir is threaded through fetch_prism → _fetch_single_day."""

    def test_cache_dir_passed_to_fetch_single_day(self, bounds, tmp_path):
        from met_timeseries.sources.prism import fetch_prism

        cache_dir = tmp_path / "prism_cache"
        mock_fn = MagicMock(side_effect=_make_mock_fetch_single_day())

        with patch("met_timeseries.sources.prism._fetch_single_day", mock_fn), \
             patch("met_timeseries.sources.prism.time.sleep"):
            fetch_prism(bounds, start="2020-06-15", variables=["ppt"], cache_dir=cache_dir)

        assert mock_fn.call_count > 0
        call_kwargs = mock_fn.call_args_list[0]
        assert call_kwargs.kwargs.get("cache_dir") == cache_dir

    def test_none_cache_dir_passed_to_fetch_single_day(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        mock_fn = MagicMock(side_effect=_make_mock_fetch_single_day())

        with patch("met_timeseries.sources.prism._fetch_single_day", mock_fn), \
             patch("met_timeseries.sources.prism.time.sleep"):
            fetch_prism(bounds, start="2020-06-15", variables=["ppt"], cache_dir=None)

        assert mock_fn.call_count > 0
        call_kwargs = mock_fn.call_args_list[0]
        assert call_kwargs.kwargs.get("cache_dir") is None


# ---------------------------------------------------------------------------
# Tests for fetch_prism resolution validation
# ---------------------------------------------------------------------------

class TestFetchPrismResolution:
    """Test resolution parameter validation in fetch_prism."""

    def test_default_resolution_is_4km(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-06-15")

        assert isinstance(ds, xr.Dataset)

    def test_valid_resolution_800m(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ), patch("met_timeseries.sources.prism.time.sleep"):
            ds = fetch_prism(bounds, start="2020-06-15", resolution="800m")

        assert isinstance(ds, xr.Dataset)

    def test_invalid_resolution_raises(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        with pytest.raises(ValueError):
            fetch_prism(bounds, start="2020-06-15", resolution="400m")


# ---------------------------------------------------------------------------
# Tests for sleep between requests
# ---------------------------------------------------------------------------

class TestSleepBetweenRequests:
    """Test that time.sleep is called between day requests."""

    def test_sleep_called_between_requests(self, bounds):
        from met_timeseries.sources.prism import fetch_prism

        mock_fetch = MagicMock(side_effect=_make_mock_fetch_single_day())
        mock_sleep = MagicMock()

        with patch("met_timeseries.sources.prism._fetch_single_day", mock_fetch), \
             patch("met_timeseries.sources.prism.time.sleep", mock_sleep):
            fetch_prism(
                bounds, start="2020-06-01", end="2020-06-03", variables=["ppt"]
            )

        assert mock_sleep.call_count >= 1
        mock_sleep.assert_any_call(2)


# ---------------------------------------------------------------------------
# Tests for _fetch_single_day coordinate handling
# ---------------------------------------------------------------------------

class TestFetchSingleDayCoordNormalization:
    """Test that y/x → lat/lon renaming and band squeezing work correctly."""

    def _make_open_da_mock(self, clip_result: xr.DataArray) -> MagicMock:
        """Return a MagicMock for xr.open_dataarray whose .rio.clip_box returns *clip_result*."""
        mock_da = MagicMock()
        mock_rio = MagicMock()
        mock_rio.clip_box.return_value = clip_result
        mock_da.rio = mock_rio
        return mock_da

    def test_renames_y_x_to_lat_lon(self, bounds, tmp_path):
        """DataArrays with y/x dims should be renamed to lat/lon."""
        from met_timeseries.sources.prism import _fetch_single_day

        lats = np.array([45.5, 45.0])
        lons = np.array([-110.0, -109.5, -109.0])
        data = np.full((len(lats), len(lons)), 2.5, dtype=np.float32)
        da_yx = xr.DataArray(
            data, dims=["y", "x"], coords={"y": lats, "x": lons}, name="ppt"
        )

        with (
            patch("met_timeseries.sources.prism._download_prism_zip", return_value=tmp_path / "z.zip"),
            patch("met_timeseries.sources.prism._extract_tif", return_value=tmp_path / "f.tif"),
            patch("xarray.open_dataarray", return_value=self._make_open_da_mock(da_yx)),
            patch.dict(sys.modules, {"rioxarray": MagicMock()}),
        ):
            result = _fetch_single_day(bounds, datetime.date(2020, 6, 15), "ppt")

        assert "lat" in result.dims
        assert "lon" in result.dims
        assert "y" not in result.dims
        assert "x" not in result.dims

    def test_squeezes_band_dimension(self, bounds, tmp_path):
        """DataArrays with a band dim should have it removed."""
        from met_timeseries.sources.prism import _fetch_single_day

        lats = np.array([45.5, 45.0])
        lons = np.array([-110.0, -109.5, -109.0])
        data = np.full((1, len(lats), len(lons)), 3.0, dtype=np.float32)
        da_with_band = xr.DataArray(
            data,
            dims=["band", "lat", "lon"],
            coords={"band": [1], "lat": lats, "lon": lons},
            name="ppt",
        )

        with (
            patch("met_timeseries.sources.prism._download_prism_zip", return_value=tmp_path / "z.zip"),
            patch("met_timeseries.sources.prism._extract_tif", return_value=tmp_path / "f.tif"),
            patch("xarray.open_dataarray", return_value=self._make_open_da_mock(da_with_band)),
            patch.dict(sys.modules, {"rioxarray": MagicMock()}),
        ):
            result = _fetch_single_day(bounds, datetime.date(2020, 6, 15), "ppt")

        assert "band" not in result.dims
        assert result.dims == ("lat", "lon")

    def test_band_and_yx_together(self, bounds, tmp_path):
        """DataArrays with both band and y/x dims should come out as (lat, lon)."""
        from met_timeseries.sources.prism import _fetch_single_day

        lats = np.array([45.5, 45.0])
        lons = np.array([-110.0, -109.5, -109.0])
        data = np.full((1, len(lats), len(lons)), 5.0, dtype=np.float32)
        da_band_yx = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={"band": [1], "y": lats, "x": lons},
            name="tmax",
        )

        with (
            patch("met_timeseries.sources.prism._download_prism_zip", return_value=tmp_path / "z.zip"),
            patch("met_timeseries.sources.prism._extract_tif", return_value=tmp_path / "f.tif"),
            patch("xarray.open_dataarray", return_value=self._make_open_da_mock(da_band_yx)),
            patch.dict(sys.modules, {"rioxarray": MagicMock()}),
        ):
            result = _fetch_single_day(bounds, datetime.date(2020, 6, 15), "tmax")

        assert "band" not in result.dims
        assert "lat" in result.dims
        assert "lon" in result.dims


# ---------------------------------------------------------------------------
# Tests for _download_prism_zip
# ---------------------------------------------------------------------------

class TestDownloadPrismZip:
    def test_raises_runtime_error_on_failure(self, tmp_path):
        from met_timeseries.sources.prism import _download_prism_zip

        with patch(
            "urllib.request.urlretrieve",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download"):
                _download_prism_zip("http://example.com/bad.zip", tmp_path)

    def test_returns_path_on_success(self, tmp_path):
        from met_timeseries.sources.prism import _download_prism_zip

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"fake zip content")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            result = _download_prism_zip("http://example.com/ok.zip", tmp_path)

        assert result.exists()
        assert result.suffix == ".zip"

    def test_no_cache_dir_saves_to_dest_dir(self, tmp_path):
        from met_timeseries.sources.prism import _download_prism_zip

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"fake zip content")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            result = _download_prism_zip(
                "http://example.com/ok.zip",
                tmp_path,
                variable="ppt",
                resolution="4km",
                date="20200615",
                cache_dir=None,
            )

        assert result.parent == tmp_path
        assert result.name == "prism_data.zip"

    def test_cache_dir_saves_with_expected_filename(self, tmp_path):
        from met_timeseries.sources.prism import _download_prism_zip

        cache_dir = tmp_path / "cache"

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"fake zip content")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            result = _download_prism_zip(
                "http://example.com/ok.zip",
                tmp_path,
                variable="ppt",
                resolution="4km",
                date="20200615",
                cache_dir=cache_dir,
            )

        assert result.parent == cache_dir
        assert result.name == "prism_ppt_4km_20200615.zip"
        assert result.exists()

    def test_cache_dir_created_if_not_exists(self, tmp_path):
        from met_timeseries.sources.prism import _download_prism_zip

        cache_dir = tmp_path / "new" / "nested" / "cache"
        assert not cache_dir.exists()

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"fake zip content")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            _download_prism_zip(
                "http://example.com/ok.zip",
                tmp_path,
                variable="tmin",
                resolution="800m",
                date="20210101",
                cache_dir=cache_dir,
            )

        assert cache_dir.exists()

    def test_cache_dir_accepts_str(self, tmp_path):
        from met_timeseries.sources.prism import _download_prism_zip

        cache_dir = str(tmp_path / "cache_str")

        def fake_retrieve(url, dest):
            Path(dest).write_bytes(b"fake zip content")

        with patch("urllib.request.urlretrieve", side_effect=fake_retrieve):
            result = _download_prism_zip(
                "http://example.com/ok.zip",
                tmp_path,
                variable="tmax",
                resolution="4km",
                date="20201231",
                cache_dir=cache_dir,
            )

        assert result.name == "prism_tmax_4km_20201231.zip"
        assert result.exists()


# ---------------------------------------------------------------------------
# Tests for _extract_tif
# ---------------------------------------------------------------------------

class TestExtractTif:
    def _make_zip_with_tif(self, tmp_path: Path) -> Path:
        """Create a minimal zip containing a dummy .tif file."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("prism_data.tif", b"FAKE TIF DATA")
        return zip_path

    def test_returns_tif_path(self, tmp_path):
        from met_timeseries.sources.prism import _extract_tif

        zip_path = self._make_zip_with_tif(tmp_path)
        result = _extract_tif(zip_path)

        assert result.suffix == ".tif"
        assert result.exists()

    def test_raises_if_no_tif_in_zip(self, tmp_path):
        from met_timeseries.sources.prism import _extract_tif

        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no tif here")

        with pytest.raises(RuntimeError, match="No .tif file"):
            _extract_tif(zip_path)

    def test_extract_dir_overrides_zip_parent(self, tmp_path):
        """When extract_dir is provided, extraction goes there, not zip_path.parent."""
        from met_timeseries.sources.prism import _extract_tif

        zip_dir = tmp_path / "zipdir"
        zip_dir.mkdir()
        extract_dir = tmp_path / "extractdir"
        extract_dir.mkdir()

        zip_path = zip_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("prism_data.tif", b"FAKE TIF DATA")

        result = _extract_tif(zip_path, extract_dir=extract_dir)

        assert result.suffix == ".tif"
        assert result.exists()
        # The tif must be inside extract_dir, not zip_dir
        assert extract_dir in result.parents
        assert zip_dir not in result.parents


# ---------------------------------------------------------------------------
# Tests for get_release_date
# ---------------------------------------------------------------------------

class TestGetReleaseDate:
    """Test the get_release_date function."""

    def _mock_urlopen(self, response_dict):
        """Return a MagicMock that works as a context manager for urlopen."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_dict).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_returns_dict_with_expected_keys(self):
        from met_timeseries.sources.prism import get_release_date

        payload = {
            "data_date": "19990101",
            "release_date": "20000101",
            "element": "tmin",
            "grid_count": 1,
            "data_url": "https://example.com/data",
        }
        mock_resp = self._mock_urlopen(payload)

        with patch("met_timeseries.sources.prism.urllib.request.urlopen", return_value=mock_resp):
            result = get_release_date("tmin", "1999-01-01")

        assert isinstance(result, dict)
        for key in ("data_date", "release_date", "element", "grid_count", "data_url"):
            assert key in result

    def test_correct_url_construction(self):
        from met_timeseries.sources.prism import get_release_date

        payload = {
            "data_date": "19990101",
            "release_date": "20000101",
            "element": "tmin",
            "grid_count": 1,
            "data_url": "https://example.com/data",
        }
        mock_resp = self._mock_urlopen(payload)

        with patch("met_timeseries.sources.prism.urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            get_release_date("tmin", "1999-01-01")

        called_url = mock_urlopen.call_args[0][0]
        assert "releaseDate/us/4km/tmin/19990101" in called_url
        assert called_url.endswith("?json=true")

    def test_invalid_resolution_raises(self):
        from met_timeseries.sources.prism import get_release_date

        with pytest.raises(ValueError):
            get_release_date("tmin", "1999-01-01", resolution="400m")

    def test_invalid_variable_raises(self):
        from met_timeseries.sources.prism import get_release_date

        with pytest.raises(ValueError):
            get_release_date("invalid_var", "1999-01-01")

    def test_raises_runtime_error_on_http_failure(self):
        from met_timeseries.sources.prism import get_release_date

        with patch(
            "met_timeseries.sources.prism.urllib.request.urlopen",
            side_effect=OSError("connection refused"),
        ):
            with pytest.raises(RuntimeError):
                get_release_date("tmin", "1999-01-01")


# ---------------------------------------------------------------------------
# Tests for AVAILABLE_VARIABLES
# ---------------------------------------------------------------------------

class TestAvailableVariables:
    def test_contains_expected_variables(self):
        from met_timeseries.sources.prism import AVAILABLE_VARIABLES

        for var in ("ppt", "tmax", "tmin", "tmean", "tdmean", "vpdmin", "vpdmax"):
            assert var in AVAILABLE_VARIABLES
