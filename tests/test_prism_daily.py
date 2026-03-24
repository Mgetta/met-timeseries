"""
Tests for met_timeseries.sources.prism — daily PRISM implementation.

All HTTP downloads are mocked so the tests run without network access.
"""

from __future__ import annotations

import datetime
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
    to mimic how rioxarray opens a BIL file.
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

    def _side_effect(bounds, date, variable):
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
# Tests for fetch_prism_daily
# ---------------------------------------------------------------------------

class TestFetchPrismDailyInterface:
    """Test the public fetch_prism_daily function."""

    def test_returns_dataset(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-06-15")

        assert isinstance(ds, xr.Dataset)

    def test_single_day_has_one_time_step(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-06-15")

        assert "time" in ds.dims
        assert ds.sizes["time"] == 1

    def test_multi_day_range(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-06-01", end="2020-06-05")

        assert ds.sizes["time"] == 5

    def test_end_same_as_start_gives_one_day(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-01-10", end="2020-01-10")

        assert ds.sizes["time"] == 1

    def test_has_lat_lon_dims(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-06-01", end="2020-06-03")

        assert "lat" in ds.dims
        assert "lon" in ds.dims

    def test_time_coords_are_datetime64(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-03-01", end="2020-03-03")

        times = ds.coords["time"].values
        assert np.issubdtype(times.dtype, np.datetime64)

    def test_time_coords_match_requested_dates(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-03-01", end="2020-03-03")

        times = ds.coords["time"].values.astype("datetime64[D]").astype(str).tolist()
        assert times == ["2020-03-01", "2020-03-02", "2020-03-03"]

    def test_default_variables(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(bounds, start="2020-06-15")

        assert set(ds.data_vars) == {"ppt", "tmax", "tmin"}

    def test_custom_variables(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with patch(
            "met_timeseries.sources.prism._fetch_single_day",
            side_effect=_make_mock_fetch_single_day(),
        ):
            ds = fetch_prism_daily(
                bounds, start="2020-06-15", variables=["ppt", "tmean"]
            )

        assert set(ds.data_vars) == {"ppt", "tmean"}

    def test_raises_value_error_when_end_before_start(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        with pytest.raises(ValueError, match="end.*before start"):
            fetch_prism_daily(bounds, start="2020-06-15", end="2020-06-10")

    def test_calls_fetch_single_day_for_each_day_and_variable(self, bounds):
        from met_timeseries.sources.prism import fetch_prism_daily

        mock_fn = MagicMock(side_effect=_make_mock_fetch_single_day())

        with patch("met_timeseries.sources.prism._fetch_single_day", mock_fn):
            fetch_prism_daily(
                bounds, start="2020-01-01", end="2020-01-03", variables=["ppt", "tmax"]
            )

        # 3 days × 2 variables = 6 calls
        assert mock_fn.call_count == 6


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
            patch("met_timeseries.sources.prism._extract_bil", return_value=tmp_path / "f.bil"),
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
            patch("met_timeseries.sources.prism._extract_bil", return_value=tmp_path / "f.bil"),
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
            patch("met_timeseries.sources.prism._extract_bil", return_value=tmp_path / "f.bil"),
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


# ---------------------------------------------------------------------------
# Tests for _extract_bil
# ---------------------------------------------------------------------------

class TestExtractBil:
    def _make_zip_with_bil(self, tmp_path: Path) -> Path:
        """Create a minimal zip containing a dummy .bil file."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("prism_data.bil", b"FAKE BIL DATA")
        return zip_path

    def test_returns_bil_path(self, tmp_path):
        from met_timeseries.sources.prism import _extract_bil

        zip_path = self._make_zip_with_bil(tmp_path)
        result = _extract_bil(zip_path)

        assert result.suffix == ".bil"
        assert result.exists()

    def test_raises_if_no_bil_in_zip(self, tmp_path):
        from met_timeseries.sources.prism import _extract_bil

        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no bil here")

        with pytest.raises(RuntimeError, match="No .bil file"):
            _extract_bil(zip_path)


# ---------------------------------------------------------------------------
# Tests for AVAILABLE_VARIABLES
# ---------------------------------------------------------------------------

class TestAvailableVariables:
    def test_contains_expected_variables(self):
        from met_timeseries.sources.prism import AVAILABLE_VARIABLES

        for var in ("ppt", "tmax", "tmin", "tmean", "tdmean", "vpdmin", "vpdmax"):
            assert var in AVAILABLE_VARIABLES
