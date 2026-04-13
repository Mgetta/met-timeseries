"""
Synoptic (MesoWest) station observation module.

Wraps the ``SynopticPy`` package (``synoptic`` on PyPI) to search for weather
stations and retrieve hourly timeseries from the Synoptic Weather API.
Covers ASOS, AWOS, RAWS, MnDOT RWIS, MN Ag Network, and dozens of other
networks.

Authentication
--------------
Requires a free Synoptic API token set via the ``SYNOPTIC_TOKEN`` environment
variable or ``~/.config/SynopticPy/config.toml``.
Register at https://customer.synopticdata.com/signup/

Lazy imports
------------
``synoptic.services`` is imported inside each function so the rest of the
package remains importable even when ``SynopticPy`` is not installed.
"""

from __future__ import annotations

import datetime as dt
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from met_timeseries.sources.base import BoundingBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minnesota-relevant Synoptic network short-names.
MN_NETWORKS: dict[str, str] = {
    "NWS/FAA": "ASOS/AWOS airport stations",
    "MN_DOT": "MnDOT RWIS road weather stations",
    "MDAG": "MN Agricultural Network",
    "RAWS": "Remote Automated Weather Stations",
}

#: Default meteorological variables requested from the Synoptic API.
HOURLY_VARIABLES: list[str] = [
    "air_temp",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "wind_gust",
    "precip_accum_one_hour",
    "solar_radiation",
    "pressure",
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def search_stations(
    bounds: BoundingBox,
    networks: list[str] | None = None,
    variables: list[str] | None = None,
    active_only: bool = True,
) -> gpd.GeoDataFrame:
    """Search for weather stations within a bounding box.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    networks:
        Optional list of Synoptic network short-names to filter by (e.g.
        ``["NWS/FAA", "MN_DOT"]``).  Defaults to all networks.
    variables:
        Optional list of Synoptic variable names to filter stations by
        (only stations reporting those variables are returned).
    active_only:
        If ``True`` (default), only return currently active stations.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``stid``, ``name``, ``latitude``, ``longitude``,
        ``elevation``, ``network``, ``mnet_id``, ``geometry``.
        CRS: EPSG:4326.
    """
    from synoptic.services import Metadata  # lazy import

    kwargs: dict = {
        "bbox": [bounds.south, bounds.west, bounds.north, bounds.east],
        "verbose": False,
    }
    if networks is not None:
        kwargs["network"] = ",".join(networks)
    if variables is not None:
        kwargs["vars"] = variables
    if active_only:
        kwargs["status"] = "active"

    logger.debug("Querying Synoptic Metadata with kwargs=%s", kwargs)
    response = Metadata(**kwargs)
    df = response.df().to_pandas()

    gdf = gpd.GeoDataFrame(
        {
            "stid": df["stid"],
            "name": df["name"],
            "latitude": df["latitude"].astype(float),
            "longitude": df["longitude"].astype(float),
            "elevation": df["elevation"].astype(float),
            "network": df["mnet_shortname"],
            "mnet_id": df["mnet_id"],
        },
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(df["longitude"].astype(float), df["latitude"].astype(float))
        ],
        crs="EPSG:4326",
    )

    return gdf


def fetch_hourly(
    station_ids: str | list[str],
    start: str,
    end: str,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch hourly observations for one or more stations.

    Parameters
    ----------
    station_ids:
        Single station ID string or a list of station ID strings (e.g.
        ``"KMSN"`` or ``["KMSN", "KMSP"]``).
    start:
        Start datetime string accepted by the Synoptic API (e.g.
        ``"202401010000"`` or ISO format ``"2024-01-01T00:00"``).
    end:
        End datetime string (same format as *start*).
    variables:
        Variables to retrieve.  Defaults to :data:`HOURLY_VARIABLES`.

    Returns
    -------
    pandas.DataFrame
        Observations with a UTC ``date_time`` index and lowercase column names.
        No caching is performed.
    """
    from synoptic.services import TimeSeries  # lazy import

    if isinstance(station_ids, str):
        station_ids = [station_ids]
    if variables is None:
        variables = HOURLY_VARIABLES

    logger.debug(
        "Fetching hourly TimeSeries for stations=%s start=%s end=%s",
        station_ids,
        start,
        end,
    )
    response = TimeSeries(
        stid=station_ids,
        start=start,
        end=end,
        vars=variables,
        verbose=False,
    )
    df = response.df().to_pandas()
    df.columns = [c.lower() for c in df.columns]
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    df = df.set_index("date_time")
    return df


def fetch_precipitation(
    station_ids: str | list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch hourly precipitation intervals for one or more stations.

    Uses the Synoptic ``Precipitation`` service with ``pmode="intervals"``
    and ``interval="hour"`` to return gauge-measured hourly precipitation
    accumulations.

    Parameters
    ----------
    station_ids:
        Single station ID string or a list of station ID strings.
    start:
        Start datetime string accepted by the Synoptic API.
    end:
        End datetime string (same format as *start*).

    Returns
    -------
    pandas.DataFrame
        Hourly precipitation with a UTC ``date_time`` index and lowercase
        column names.  No caching is performed.
    """
    from synoptic.services import Precipitation  # lazy import

    if isinstance(station_ids, str):
        station_ids = [station_ids]

    logger.debug(
        "Fetching Precipitation for stations=%s start=%s end=%s",
        station_ids,
        start,
        end,
    )
    response = Precipitation(
        stid=station_ids,
        start=start,
        end=end,
        pmode="intervals",
        interval="hour",
        verbose=False,
    )
    df = response.df().to_pandas()
    df.columns = [c.lower() for c in df.columns]
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    df = df.set_index("date_time")
    return df
