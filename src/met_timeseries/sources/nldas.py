"""NLDAS-2 data source wrapping pynldas2."""

from __future__ import annotations

import calendar
import logging

import geopandas as gpd
import xarray as xr
from shapely.geometry import box

from met_timeseries.sources.base import DataSource

logger = logging.getLogger(__name__)

#: All raw variables available from the NLDAS-2 forcing file A.
NLDAS_VARIABLES = [
    "prcp",      # Precipitation hourly total (kg/m2)
    "rsds",      # Downward shortwave radiation (W/m2)
    "rlds",      # Downward longwave radiation (W/m2)
    "temp",      # Air temperature at 2m (K)
    "humidity",  # Specific humidity at 2m (kg/kg)
    "wind_u",    # Eastward wind component at 10m (m/s)
    "wind_v",    # Northward wind component at 10m (m/s)
    "psurf",     # Surface pressure (Pa)
    "pet",       # Potential evapotranspiration (kg/m2)
]


class NLDASSource(DataSource):
    """NLDAS-2 hourly forcing data source.

    Wraps :func:`pynldas2.get_bygeom` to fetch data for the bounding box
    of the provided polygon layer.  Data are fetched for the full calendar
    month specified by *year* and *month*.

    Parameters
    ----------
    variables : list[str] or None
        Subset of :data:`NLDAS_VARIABLES` to fetch.  Defaults to all
        available variables.
    """

    def __init__(self, variables: list[str] | None = None) -> None:
        self._variables = variables or NLDAS_VARIABLES

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def fetch(self, polygons: gpd.GeoDataFrame, year: int, month: int) -> xr.Dataset:
        """Fetch NLDAS-2 hourly data for the polygon bounding box.

        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            Source polygons; their total bounding box is used as the
            spatial extent.
        year : int
            Four-digit year.
        month : int
            Month number (1-12).

        Returns
        -------
        xr.Dataset
            Hourly NLDAS-2 Dataset for the requested period and bounding
            box.
        """
        try:
            import pynldas2 as nldas  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pynldas2 is required to use NLDASSource. "
                "Install it with: pip install pynldas2"
            ) from exc

        # Build date strings for the full calendar month
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{last_day}"

        # Get bounding box in EPSG:4326
        poly_4326 = polygons.to_crs("EPSG:4326") if polygons.crs.to_epsg() != 4326 else polygons
        minx, miny, maxx, maxy = poly_4326.total_bounds
        geometry = box(minx, miny, maxx, maxy)

        logger.info(
            "Fetching NLDAS-2 data for %s-%02d (bbox: %.2f,%.2f,%.2f,%.2f)",
            year, month, minx, miny, maxx, maxy,
        )

        ds = nldas.get_bygeom(geometry, dates=(start, end), source="grib")
        return ds

    def available_variables(self) -> list[str]:
        """Return the list of NLDAS-2 variable names.

        Returns
        -------
        list[str]
        """
        return list(NLDAS_VARIABLES)

    def temporal_resolution(self) -> str:
        """Return the temporal resolution of NLDAS-2 data.

        Returns
        -------
        str
            ``"1h"``
        """
        return "1h"

    def spatial_resolution(self) -> str:
        """Return the nominal spatial resolution of NLDAS-2 data.

        Returns
        -------
        str
            ``"0.125 deg"``
        """
        return "0.125 deg"
