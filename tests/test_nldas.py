"""
Tests for met_timeseries.sources.nldas — URL template and error handling.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


class TestOpendapTemplate:
    """Verify that the URL template produces correct filenames."""

    def test_url_ends_with_grb(self) -> None:
        from met_timeseries.sources.nldas import _OPENDAP_TEMPLATE

        url = _OPENDAP_TEMPLATE.format(year=2010, doy=1, month=1, day=1, hour=0)
        assert url.endswith(".grb"), f"URL should end with .grb, got: {url}"

    def test_url_does_not_contain_sub_nc(self) -> None:
        from met_timeseries.sources.nldas import _OPENDAP_TEMPLATE

        url = _OPENDAP_TEMPLATE.format(year=2010, doy=1, month=1, day=1, hour=0)
        assert ".SUB.nc" not in url, f"URL must not contain .SUB.nc suffix: {url}"

    def test_url_uses_version_020(self) -> None:
        from met_timeseries.sources.nldas import _OPENDAP_TEMPLATE

        url = _OPENDAP_TEMPLATE.format(year=2010, doy=1, month=1, day=1, hour=0)
        assert ".020.grb" in url, f"URL should contain version code .020, got: {url}"

    def test_url_does_not_use_version_002(self) -> None:
        from met_timeseries.sources.nldas import _OPENDAP_TEMPLATE

        url = _OPENDAP_TEMPLATE.format(year=2010, doy=1, month=1, day=1, hour=0)
        assert ".002." not in url, f"URL must not contain old version code .002: {url}"

    def test_url_example_matches_expected(self) -> None:
        from met_timeseries.sources.nldas import _OPENDAP_TEMPLATE

        url = _OPENDAP_TEMPLATE.format(year=2010, doy=1, month=1, day=1, hour=0)
        expected = (
            "https://hydro1.gesdisc.eosdis.nasa.gov/opendap/NLDAS/NLDAS_FORA0125_H.2.0/"
            "2010/001/NLDAS_FORA0125_H.A20100101.0000.020.grb"
        )
        assert url == expected


@pytest.fixture()
def bounds():
    from met_timeseries.sources.base import BoundingBox

    return BoundingBox(west=-110.0, east=-109.0, south=45.0, north=46.0)


class TestOpenDatasetErrorHandling:
    """Verify that _open_dataset raises descriptive RuntimeErrors for HTTP errors."""

    @pytest.mark.parametrize(
        "error_msg, exc_type, match",
        [
            ("404 not found", RuntimeError, "File not found"),
            ("401 unauthorized", RuntimeError, "Authentication required"),
            ("403 forbidden", RuntimeError, "Access forbidden"),
            ("connection reset", OSError, "connection reset"),
        ],
    )
    def test_http_errors_produce_correct_exception(
        self, bounds, error_msg: str, exc_type: type, match: str
    ) -> None:
        from met_timeseries.sources.nldas import _open_dataset

        with patch("xarray.open_dataset", side_effect=OSError(error_msg)):
            with pytest.raises(exc_type, match=match):
                _open_dataset("http://example.com/fake.grb", bounds, ["APCP"], None)
