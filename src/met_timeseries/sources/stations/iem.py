"""
Iowa Environmental Mesonet (IEM) station observation data source.

Downloads hourly ASOS/AWOS and RWIS observations as CSV over public HTTP.
No API key, account, or authentication required.

Intended for validation — comparing pipeline outputs against observed data.

Archive: https://mesonet.agron.iastate.edu/
ASOS docs: https://mesonet.agron.iastate.edu/request/download.phtml
RWIS docs: https://mesonet.agron.iastate.edu/request/rwis/fe.phtml
"""

from __future__ import annotations

import json
import logging
import time
from io import StringIO
from urllib.request import Request, urlopen

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from met_timeseries.sources.base import BoundingBox

logger = logging.getLogger(__name__)

_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
_RWIS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/rwis.py"
_NETWORK_GEOJSON_URL = (
    "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
)

MN_NETWORKS = ["MN_ASOS", "MN_RWIS"]

# IEM ASOS variable codes and human-readable names
ASOS_VARIABLES = {
    "tmpf": "air_temp_f",
    "dwpf": "dewpoint_f",
    "relh": "relative_humidity",
    "sknt": "wind_speed_kt",
    "drct": "wind_direction_deg",
    "gust": "wind_gust_kt",
    "p01i": "precip_1hr_in",
    "mslp": "pressure_mb",
    "alti": "altimeter_inhg",
    "vsby": "visibility_mi",
    "skyc1": "sky_cover_1",
    "feel": "feels_like_f",
}

# IEM RWIS atmospheric variables
RWIS_VARIABLES = {
    "tmpf": "air_temp_f",
    "dwpf": "dewpoint_f",
    "relh": "relative_humidity",
    "sknt": "wind_speed_kt",
    "drct": "wind_direction_deg",
    "gust": "wind_gust_kt",
    "feel": "feels_like_f",
}

_MAX_ATTEMPTS = 6


# ---------------------------------------------------------------------------
# Station discovery
# ---------------------------------------------------------------------------


def search_stations(
    bounds: BoundingBox,
    networks: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Find weather stations within a bounding box.

    Fetches station metadata from IEM's GeoJSON network endpoint and
    filters to those within the bounding box.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.
    networks:
        IEM network codes (e.g. ``["MN_ASOS", "MN_RWIS"]``).
        Defaults to :data:`MN_NETWORKS`.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``stid``, ``name``, ``elevation``, ``network``,
        ``latitude``, ``longitude``, ``geometry``.
    """
    if networks is None:
        networks = MN_NETWORKS

    records = []
    for network in networks:
        url = _NETWORK_GEOJSON_URL.format(network=network)
        try:
            data = urlopen(url)  # noqa: S310
            jdict = json.load(data)
        except Exception as exc:
            logger.warning("Failed to fetch station list for %s: %s", network, exc)
            continue

        for site in jdict.get("features", []):
            lon = site["geometry"]["coordinates"][0]
            lat = site["geometry"]["coordinates"][1]

            if not (bounds.south <= lat <= bounds.north
                    and bounds.west <= lon <= bounds.east):
                continue

            props = site["properties"]
            records.append({
                "stid": props.get("sid", ""),
                "name": props.get("sname", ""),
                "elevation": props.get("elevation", None),
                "network": network,
                "latitude": lat,
                "longitude": lon,
                "geometry": Point(lon, lat),
            })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logger.info("Found %d stations in bounds %s", len(gdf), bounds)
    return gdf


# ---------------------------------------------------------------------------
# ASOS / AWOS hourly data
# ---------------------------------------------------------------------------


def fetch_asos(
    station_ids: str | list[str],
    start: str,
    end: str,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch hourly ASOS/AWOS observations for one or more stations.

    Parameters
    ----------
    station_ids:
        Station identifier(s) — 3-letter FAA code (e.g. ``"MSP"``) or
        4-letter ICAO code (e.g. ``"KMSP"``).  IEM accepts both.
    start:
        Start date in ``"YYYY-MM-DD"`` format.
    end:
        End date in ``"YYYY-MM-DD"`` format (inclusive).
    variables:
        IEM ASOS variable codes (keys of :data:`ASOS_VARIABLES`).
        Defaults to all.

    Returns
    -------
    pandas.DataFrame
        Index: ``valid`` (UTC datetime).
        Columns: ``station``, plus one renamed column per variable.
    """
    if isinstance(station_ids, str):
        station_ids = [station_ids]
    if variables is None:
        variables = list(ASOS_VARIABLES.keys())

    # Build URL — IEM expects repeated `data=` and `station=` params
    parts = [
        f"station={stid}" for stid in station_ids
    ] + [
        f"data={var}" for var in variables
    ] + [
        "tz=Etc/UTC",
        "format=onlycomma",
        "latlon=yes",
        "missing=empty",
        "trace=0.0001",
        f"year1={start[:4]}",
        f"month1={int(start[5:7])}",
        f"day1={int(start[8:10])}",
        f"year2={end[:4]}",
        f"month2={int(end[5:7])}",
        f"day2={int(end[8:10])}",
    ]

    url = f"{_ASOS_URL}?{'&'.join(parts)}"
    text = _download(url)

    if not text or text.startswith("ERROR") or "no data" in text.lower():
        logger.warning("No ASOS data returned for %s", station_ids)
        return pd.DataFrame()

    df = pd.read_csv(StringIO(text), na_values=["M", ""])
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    df = df.set_index("valid").sort_index()

    # rename variable columns to human-readable names
    df = df.rename(columns=ASOS_VARIABLES)

    return df


# ---------------------------------------------------------------------------
# RWIS hourly data
# ---------------------------------------------------------------------------


def fetch_rwis(
    station_ids: str | list[str],
    start: str,
    end: str,
    network: str = "MN_RWIS",
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch hourly RWIS (road weather) observations.

    Parameters
    ----------
    station_ids:
        RWIS station identifier(s) (e.g. ``"RALN5"``).
        Use ``"_ALL"`` to fetch all stations in the network.
    start:
        Start date in ``"YYYY-MM-DD"`` format.
    end:
        End date in ``"YYYY-MM-DD"`` format (inclusive).
    network:
        IEM RWIS network code. Defaults to ``"MN_RWIS"``.
    variables:
        RWIS variable codes (keys of :data:`RWIS_VARIABLES`).
        Defaults to all atmospheric variables.

    Returns
    -------
    pandas.DataFrame
        Index: ``obtime`` (UTC datetime).
        Columns: ``station``, plus one renamed column per variable.
    """
    if isinstance(station_ids, str):
        station_ids = [station_ids]
    if variables is None:
        variables = list(RWIS_VARIABLES.keys())

    stations_param = "&".join(f"stations={s}" for s in station_ids)
    vars_param = ",".join(variables)

    url = (
        f"{_RWIS_URL}?"
        f"network={network}"
        f"&{stations_param}"
        f"&tz=UTC"
        f"&what=download"
        f"&src=atmos"
        f"&vars={vars_param}"
        f"&sts={start}T00:00"
        f"&ets={end}T23:59"
    )

    text = _download(url)

    if not text or "no results" in text.lower() or "sorry" in text.lower():
        logger.warning("No RWIS data returned for %s", station_ids)
        return pd.DataFrame()

    df = pd.read_csv(StringIO(text), na_values=["M", ""])

    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    time_col = "obtime" if "obtime" in df.columns else "valid"
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.rename(columns={time_col: "valid"})
        df = df.set_index("valid").sort_index()

    df = df.rename(columns=RWIS_VARIABLES)

    return df


# ---------------------------------------------------------------------------
# Download helper with retry
# ---------------------------------------------------------------------------


def _download(url: str) -> str:
    """Fetch text data from IEM with exponential backoff."""
    logger.debug("Downloading: %s", url)
    
    # IEM requires a custom User-Agent to avoid being blocked
    headers = {"User-Agent": "met-timeseries/1.0 (contact: soli@email.com)"}
    
    for attempt in range(_MAX_ATTEMPTS):
        try:
            req = Request(url, headers=headers)
            data = urlopen(req, timeout=300).read().decode("utf-8")
            if data and not data.startswith("ERROR"):
                return data
        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt + 1, _MAX_ATTEMPTS, url, exc,
            )
        time.sleep(2 ** attempt)

    logger.error("All %d download attempts failed for %s", _MAX_ATTEMPTS, url)
    return ""