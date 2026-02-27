"""Abstract base class for met-timeseries data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

import geopandas as gpd
import xarray as xr


class DataSource(ABC):
    """Abstract interface for gridded meteorological data sources.

    Subclasses must implement :meth:`fetch`, :meth:`available_variables`,
    :meth:`temporal_resolution`, and :meth:`spatial_resolution`.
    """

    @abstractmethod
    def fetch(self, polygons: gpd.GeoDataFrame, year: int, month: int) -> xr.Dataset:
        """Fetch gridded data for the bounding box of *polygons*.

        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            GeoDataFrame whose total bounding box defines the spatial extent
            to retrieve.
        year : int
            Four-digit year.
        month : int
            Month number (1-12).

        Returns
        -------
        xr.Dataset
            Dataset containing one or more data variables on a regular grid,
            with an ``x``/``lon`` and ``y``/``lat`` dimension and a ``time``
            dimension for time-varying data.
        """

    @abstractmethod
    def available_variables(self) -> list[str]:
        """Return the list of variable names provided by this source.

        Returns
        -------
        list[str]
            Variable names as they appear in the returned :class:`xr.Dataset`.
        """

    @abstractmethod
    def temporal_resolution(self) -> str:
        """Describe the temporal resolution of the source data.

        Returns
        -------
        str
            Human-readable string such as ``"1h"`` or ``"1D"``.
        """

    @abstractmethod
    def spatial_resolution(self) -> str:
        """Describe the nominal spatial resolution of the source data.

        Returns
        -------
        str
            Human-readable string such as ``"0.125 deg"`` or ``"4 km"``.
        """
