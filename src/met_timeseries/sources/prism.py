"""PRISM daily precipitation data source."""

from __future__ import annotations

import calendar
import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
import xarray as xr

from met_timeseries.sources.base import DataSource

logger = logging.getLogger(__name__)

#: Base URL template for PRISM BIL files.
PRISM_URL_TEMPLATE = (
    "https://services.nacse.org/prism/data/public/4km/ppt/{year}{month:02d}{day:02d}"
)


class PRISMSource(DataSource):
    """PRISM 4 km daily precipitation data source.

    Downloads PRISM BIL zip archives from the PRISM web service and caches
    them locally.  Only daily precipitation is currently supported.

    Parameters
    ----------
    cache_dir : Path or str
        Directory for caching downloaded PRISM files.
        Defaults to ``"cache/prism"``.
    timeout : int
        HTTP request timeout in seconds. Default is ``120``.
    """

    def __init__(
        self,
        cache_dir: Path | str = "cache/prism",
        timeout: int = 120,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def fetch(self, polygons: gpd.GeoDataFrame, year: int, month: int) -> xr.Dataset:
        """Download and parse daily PRISM precipitation for a full month.

        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            Source polygons; used only for spatial subsetting metadata.
        year : int
            Four-digit year.
        month : int
            Month number (1-12).

        Returns
        -------
        xr.Dataset
            Daily precipitation Dataset with variable ``prcp`` (mm/day),
            dimensions ``(time, lat, lon)``.
        """
        last_day = calendar.monthrange(year, month)[1]
        daily_arrays = []
        dates = []
        meta = {}

        for day in range(1, last_day + 1):
            date_str = f"{year}{month:02d}{day:02d}"
            bil_path = self._ensure_cached(year, month, day)
            if bil_path is None:
                logger.warning("Skipping PRISM day %s (download failed)", date_str)
                continue
            arr, meta = _read_bil(bil_path)
            daily_arrays.append(arr)
            dates.append(f"{year}-{month:02d}-{day:02d}")

        if not daily_arrays:
            raise RuntimeError(f"No PRISM data could be downloaded for {year}-{month:02d}")

        # Build coordinate arrays from the last-read metadata
        nrows = int(meta["nrows"])
        ncols = int(meta["ncols"])
        xllcorner = float(meta["xllcorner"])
        yllcorner = float(meta["yllcorner"])
        cellsize = float(meta["cellsize"])

        lons = xllcorner + cellsize * (np.arange(ncols) + 0.5)
        lats = yllcorner + cellsize * (nrows - np.arange(nrows) - 0.5)

        data = np.stack(daily_arrays, axis=0)  # (time, lat, lon)
        ds = xr.Dataset(
            {"prcp": (["time", "lat", "lon"], data)},
            coords={
                "time": np.array(dates, dtype="datetime64[D]"),
                "lat": lats,
                "lon": lons,
            },
        )
        ds["prcp"].attrs["units"] = "mm/day"
        return ds

    def available_variables(self) -> list[str]:
        """Return the list of variables provided by the PRISM source.

        Returns
        -------
        list[str]
            ``["prcp"]``
        """
        return ["prcp"]

    def temporal_resolution(self) -> str:
        """Return the temporal resolution of PRISM data.

        Returns
        -------
        str
            ``"1D"``
        """
        return "1D"

    def spatial_resolution(self) -> str:
        """Return the nominal spatial resolution of PRISM data.

        Returns
        -------
        str
            ``"4 km"``
        """
        return "4 km"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_cached(self, year: int, month: int, day: int) -> Path | None:
        """Return path to the cached BIL file, downloading if necessary.

        Parameters
        ----------
        year : int
        month : int
        day : int

        Returns
        -------
        Path or None
            Path to the ``.bil`` file, or ``None`` if the download failed.
        """
        date_str = f"{year}{month:02d}{day:02d}"
        bil_path = self.cache_dir / f"PRISM_ppt_stable_4kmD2_{date_str}_bil.bil"
        if bil_path.exists():
            return bil_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        url = PRISM_URL_TEMPLATE.format(year=year, month=month, day=day)
        zip_path = self.cache_dir / f"{date_str}.zip"

        try:
            logger.debug("Downloading PRISM %s from %s", date_str, url)
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            zip_path.write_bytes(response.content)
        except requests.RequestException as exc:
            logger.error("Failed to download PRISM %s: %s", date_str, exc)
            return None

        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self.cache_dir)
        except zipfile.BadZipFile as exc:
            logger.error("Bad zip file for PRISM %s: %s", date_str, exc)
            zip_path.unlink(missing_ok=True)
            return None
        finally:
            zip_path.unlink(missing_ok=True)

        if not bil_path.exists():
            # Search for whatever BIL was extracted
            bil_files = list(self.cache_dir.glob(f"*{date_str}*.bil"))
            if bil_files:
                return bil_files[0]
            return None

        return bil_path


def _read_bil(bil_path: Path) -> tuple[np.ndarray, dict]:
    """Read a PRISM BIL raster file into a NumPy array.

    Parameters
    ----------
    bil_path : Path
        Path to the ``.bil`` binary file.  A companion ``.hdr`` file
        must exist in the same directory.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(data_array, metadata_dict)`` where *data_array* has shape
        ``(nrows, ncols)`` and *metadata_dict* contains the parsed
        header values.
    """
    hdr_path = bil_path.with_suffix(".hdr")
    meta: dict = {}
    with open(hdr_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                key, val = parts
                try:
                    meta[key.lower()] = float(val)
                except ValueError:
                    meta[key.lower()] = val

    nrows = int(meta["nrows"])
    ncols = int(meta["ncols"])
    nodata = float(meta.get("nodata", -9999.0))
    nbits = int(meta.get("nbits", 32))
    byteorder = meta.get("byteorder", "I")  # I = little-endian Intel

    fmt = "<f4" if (nbits == 32 and byteorder.upper() == "I") else ">f4"
    data = np.fromfile(bil_path, dtype=fmt).reshape(nrows, ncols)
    data = data.astype(np.float32)
    data[data == nodata] = np.nan

    return data, meta
