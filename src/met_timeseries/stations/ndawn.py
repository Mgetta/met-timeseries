"""
NDAWN (North Dakota Agricultural Weather Network) station data source.

Downloads hourly observations from the NDAWN CSV endpoint. Covers the
MN Agricultural Network (MNag) stations that are not available through IEM.

No API key, account, or authentication required.

Archive: https://ndawn.ndsu.nodak.edu/
Station map: https://ndawn.ndsu.nodak.edu/station-info.html
"""

from __future__ import annotations

import logging
import time
from io import StringIO
from urllib.request import urlopen, Request
import requests

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from met_timeseries.sources.base import BoundingBox

logger = logging.getLogger(__name__)

_STATION_INFO_URL = "https://ndawn.ndsu.nodak.edu/"
_BASE_URL = "https://ndawn.ndsu.nodak.edu/table.csv"
_TABLE_START_LINE = 3
_MAX_ATTEMPTS = 4

# MN Agricultural Network station IDs on NDAWN
# Source: https://ndawn.ndsu.nodak.edu/station-info.html (filter to MN)

# Could also be in metric not sure if I control that with the URL params
COLUMN_UNITS = {
 'Latitude': 'deg',
 'Longitude': 'deg',
 'Elevation': 'ft',
 'Hour': 'CST',
 'Avg Air Temp': 'Degrees F',
 'Avg Rel Hum': '%',
 'Avg Bare Soil Temp': 'Degrees F',
 'Avg Turf Soil Temp': 'Degrees F',
 'Avg Wind Speed': 'mph',
 'Max Wind Speed': 'mph',
 'Avg Wind Dir': 'deg',
 'Avg Wind Dir SD': 'deg',
 'Avg Sol Rad': 'Lys',
 'Total Rainfall': 'inch',
 'Avg Baro Press': 'mb',
 'Avg Dew Point': 'Degrees F',
 'Avg Wind Chill': 'Degrees F',
 'Avg Air Temp at 9 m': 'Degrees F',
 'Avg Rel Hum at 9 m': '%',
 'Avg Wind Speed at 10 m': 'mph',
 'Max Wind Speed at 10 m': 'mph',
 'Avg Wind Dir at 10 m': 'deg',
 'Avg Wind Dir SD at 10 m': 'deg'}

HOURLY_VARIABLE_MAP = {
    'hdt': 'Avg Air Temp',               
    'hdrh': 'Avg Rel Hum', 
    'hdbst': 'Avg Bare Soil Temp',
    'hdtst': 'Avg Turf Soil Temp', 
    'hdws': 'Avg Wind Speed',
    'hdmxws': 'Max Wind Speed',
    'hdwd': 'Avg Wind Dir', 
    'hdsdwd': 'Avg Wind Dir SD', 
    'hdsr': 'Avg Sol Rad',
    'hdr': 'Total Rainfall', 
    'hdbp': 'Avg Baro Press', 
    'hddp': 'Avg Dew Point',
    'hdwc': 'Avg Wind Chill', 
    'hdt9': 'Avg Air Temp at 9 m',
    'hdrh9': 'Avg Rel Hum at 9 m', 
    'hdws10': 'Avg Wind Speed at 10 m',
    'hdmxws10': 'Max Wind Speed at 10 m', 
    'hdwd10': 'Avg Wind Dir at 10 m',
    'hdsdwd10': 'Avg Wind Dir SD at 10 m'
}

STANDARD_VARIABLES = [
 'Avg Air Temp',
 'Avg Rel Hum', 
 'Avg Bare Soil Temp',
 'Avg Turf Soil Temp', 
 'Avg Wind Speed',
 'Max Wind Speed',
 'Avg Wind Dir', 
 'Avg Wind Dir SD', 
 'Avg Sol Rad',
 'Total Rainfall', 
 'Avg Baro Press', 
 'Avg Dew Point',
 'Avg Wind Chill', 
 'Avg Air Temp at 9 m',
 'Avg Rel Hum at 9 m', 
 'Avg Wind Speed at 10 m',
 'Max Wind Speed at 10 m', 
 'Avg Wind Dir at 10 m',
 'Avg Wind Dir SD at 10 m']


_METADATA_COLS = ['Station Name',
                     'Elevation']

_COORD_COLS = ['Latitude', 'Longitude',
               'time']

# https://ndawn.ndsu.nodak.edu/table.csv?station=95&
# variable=ddmxt&
# variable=ddmxtt&
# variable=ddmnt&
# variable=ddmntt&
# variable=ddavt&
# variable=dddtr&
# variable=ddbst&
# variable=ddtst&
# variable=ddws&
# variable=ddmxws&
# variable=ddmxwst&
# variable=ddwd&
# variable=ddwdsd&
# variable=ddsr&
# variable=ddtpetp&
# variable=ddtpetjh&variable=ddr&variable=dddp&variable=ddwc&variable=ddmnwc&variable=ddmxt9&variable=ddmxtt9&variable=ddmnt9&variable=ddmntt9&variable=ddmxws10&variable=ddmxwst10&variable=ddwd10&variable=ddwdsd10&dfy=&year=2026&ttype=daily&quick_pick=&begin_date=2026-04-25&end_date=2026-04-25
DAILY_VARIABLE_MAP = {
    'ddmxt': 'Max Air Temp',               
    'ddmxtt': 'Max Air Temp Time',
    'ddmnt': 'Min Air Temp', 
    'ddmntt': 'Min Air Temp Time',
    'ddavt': 'Avg Air Temp', 
    'dddtr': 'Diurnal Temp Range',
    'ddbst': 'Avg Bare Soil Temp',
    'ddtst': 'Avg Turf Soil Temp', 
    'ddws': 'Avg Wind Speed',
    'ddmxws': 'Max Wind Speed',
    'ddmxwst': 'Max Wind Speed Time',
    'ddwd': 'Avg Wind Dir', 
    'ddwdsd': 'Avg Wind Dir SD', 
    'ddsr': 'Total Solar Rad',
    'ddtpetp': 'PET Penman Daily',
    'ddtpetjh': 'PET Penman Monteith Daily',
    'ddr':  'Total Rainfall', 
    'dddp':  'Avg Baro Press', 
    'ddwc':  'Avg Wind Chill', 
    'ddmnwc':  "Min Wind Chill",
    'ddmxt9':  "Max Air Temp at 9 m",
    'ddmxtt9': "Max Air Temp at 9 m Time",
    'ddmnt9':  "Min Air Temp at 9 m", 
    'ddmntt9': "Min Air Temp at 9 m Time",
    'ddmxws10': "Max Wind Speed at 10 m",
    'ddmxwst10': "Max Wind Speed at 10 m Time", 
    'ddwd10': "Avg Wind Dir at 10 m",
    'ddwdsd10': "Avg Wind Dir SD at 10 m"
}

def _to_xarray_dataset(
    df: pd.DataFrame,
    tstep: str
) -> xr.Dataset:
    """
    Convert a DataFrame to an xarray Dataset with programmatic names and rich metadata.
    """

    if tstep == 'daily':
        variable_map = DAILY_VARIABLE_MAP
    elif tstep == 'hourly':
        variable_map = HOURLY_VARIABLE_MAP
    else:
        raise ValueError(f"Unknown time step '{tstep}'. Use 'daily' or 'hourly'.")

    # 1. Create a copy and ensure time is a datetime objects
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # 2. Set the index to your primary coordinates
    # For NDAWN, this is usually ['Station Name', 'time']
    # This allows xarray to automatically create the dimensions
    #primary_coords = [c for c in _COORD_COLS if c in df.columns]
    df = df.set_index(_COORD_COLS)
    metadata = {col: df[col].iloc[0] for col in _METADATA_COLS}
    # Filter DF to only include variables we care about
    valid_cols = [c for c in df.columns if c in variable_map.values()]
    df = df[valid_cols]
    #map df columns to short name
    rename_map = {v: k for k, v in variable_map.items() }
    df = df.rename(columns=rename_map)
    ds = df.to_xarray()

    # 4. Attach Long Names/Descriptions to individual variables
    for var_name in ds.data_vars:
        if var_name in variable_map:
            ds[var_name].attrs['long_name'] = variable_map[var_name]
            # TODO: standard units here if you have a map for them
            # ds[var_name].attrs['units'] = 'degC' 

    # 6. Global Metadata Attributes
    for name, value in metadata.items():
        ds.attrs[name] = value

    return ds

def get_stations() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of available NDAWN stations in MN."""
    
        
    import re

    response = requests.get(_STATION_INFO_URL)
    # Save the HTML snippet you provided to a variable
    html_content = response.text

    stations = []

    # 1. Use a regular expression to find all <area> tags and extract the title and href
    # The pattern looks for:
    #   <area ... title="STATION_NAME" ... href="STATION_URL">
    # We use capturing groups `(.*?)` to extract the values inside the quotes
    pattern = r'<area[^>]*title="([^"]+)"[^>]*href="([^"]+)"'

    matches = re.findall(pattern, html_content)

    for match in matches:
        station_name = match[0]
        station_url = match[1]
        
        # 2. Extract the internal ID from the URL (e.g., /station-info.html?station=78)
        id_match = re.search(r'\?station=(\d+)', station_url)
        
        if id_match:
            internal_id = id_match.group(1)
            
            stations.append({
                'name': station_name,
                'url': f"https://ndawn.ndsu.nodak.edu{station_url}",
                'internal_id': internal_id
            })

    print(f"Found {len(stations)} stations!")
    return pd.DataFrame(stations)


def get_station_data_daily(station_id,variables = None, begin_year = 1996, end_year = 2026, as_dataset = True):
    """
    Downloads daily weather data from NDAWN.
    
    Args:
        station_id (str/int): The internal ID for the station (e.g., 115).
        variables (list): List of variable codes (e.g., ['ddmxt', 'ddr']).
        begin_year (int): Start year (e.g., 1996).
        end_year (int): End year (e.g., 2026).
    """
    if variables is None:
        variables = list(DAILY_VARIABLE_MAP.keys())

    begin_date = f"{begin_year}-01-01"
    end_date = f"{end_year}-12-31"
    # Setting up the dictionary for URL arguments
    # Note: Using a list for 'variable' allows 'requests' to repeat 
    # the key in the URL (e.g., &variable=ddmxt&variable=ddr)
    params = {
        'station': station_id,
        'variable': variables,
        'ttype': 'daily',
        'quick_pick': '',
        'begin_date': begin_date,
        'end_date': end_date
    }

    print(f"Downloading data for station {station_id}...")
    response = requests.get(_BASE_URL, params=params)
    
    # Raise an exception for bad status codes (404, 500, etc.)
    response.raise_for_status()
    
    lines = response.text.splitlines()
    
    df = pd.read_csv(StringIO("\n".join(lines[_TABLE_START_LINE:])), skiprows=[1])
    
    df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    
    if as_dataset:
        df = _to_xarray_dataset(df,'daily')
    
    return df

def get_station_data_hourly(station_id, variables = None, begin_year = 1996, end_year = 2026, as_dataset = True):
    """
    Downloads weather data from NDAWN.
    
    Args:
        station_id (str/int): The internal ID for the station (e.g., 115).
        variables (list): List of variable codes (e.g., ['hdt', 'hdrh']).
        begin_year (int): Start year (e.g., 1996).
        end_year (int): End year (e.g., 2026).
    """
    if variables is None:
        variables = list(HOURLY_VARIABLE_MAP.keys())

    begin_date = f"{begin_year}-01-01"
    end_date = f"{end_year}-12-31"
    # Setting up the dictionary for URL arguments
    # Note: Using a list for 'variable' allows 'requests' to repeat 
    # the key in the URL (e.g., &variable=hdt&variable=hdrh)
    params = {
        'station': station_id,
        'variable': variables,
        'ttype': 'hourly',
        'quick_pick': '',
        'begin_date': begin_date,
        'end_date': end_date
    }

    print(f"Downloading data for station {station_id}...")
    response = requests.get(_BASE_URL, params=params)
    
    # Raise an exception for bad status codes (404, 500, etc.)
    response.raise_for_status()
    
    # NDAWN returns a CSV with some metadata at the top. 
    # We need to find where the actual data starts.
    content = response.text
    lines = content.splitlines()
    
    # # Typically, the actual headers start at a line containing 'Station Name'
    # start_line = 0
    # for i, line in enumerate(lines):
    #     if "Station Name" in line:
    #         start_line = i
    #         break
    
    # Read the cleaned CSV content into a Pandas DataFrame
    # We skip the second row (index 1 from our start) because it's usually units
    df = pd.read_csv(StringIO("\n".join(lines[_TABLE_START_LINE:])), skiprows=[1])
    
    # Convert the date and time columns into a single datetime index
    df['Hour'] = (df['Hour']/100).astype(int)-1
    df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    if as_dataset:
        df = _to_xarray_dataset(df,'hourly')
    
    return df



def search_stations(
    bounds: BoundingBox | None = None,
) -> gpd.GeoDataFrame:
    """Return MN Agricultural Network stations on NDAWN.

    Fetches a single hourly record from all MN stations to extract
    their coordinates, or returns the hardcoded station list filtered
    by bounds.

    Parameters
    ----------
    bounds:
        Spatial bounding box in EPSG:4326.  If ``None``, returns all
        MN NDAWN stations.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``station_id``, ``name``, ``latitude``, ``longitude``,
        ``elevation``, ``network``, ``geometry``.
    """
    # fetch one day of data from all MN stations to get coordinates
    station_params = "&".join(f"station={sid}" for sid in MN_STATIONS)
    url = (
        f"{_BASE_URL}?"
        f"{station_params}"
        f"&variable=hdt"
        f"&ttype=hourly"
        f"&quick_pick="
        f"&begin_date=2024-06-01"
        f"&end_date=2024-06-01"
    )

    text = _download(url)
    if not text:
        # fall back to hardcoded names without coordinates
        logger.warning("Could not fetch NDAWN station metadata, returning names only")
        records = [
            {"station_id": sid, "name": name, "network": "MN_NDAWN"}
            for sid, name in MN_STATIONS.items()
        ]
        return gpd.GeoDataFrame(records)

    df = _parse_csv(text)

    # deduplicate to one row per station
    meta_cols = ["Station Name", "Latitude", "Longitude", "Elevation"]
    # normalize column names for lookup
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    available = [c for c in meta_cols if c in df.columns]
    if not available:
        logger.warning("NDAWN CSV missing location columns")
        records = [
            {"station_id": sid, "name": name, "network": "MN_NDAWN"}
            for sid, name in MN_STATIONS.items()
        ]
        return gpd.GeoDataFrame(records)

    stations = df.drop_duplicates(subset=["Station Name"])[available].copy()

    records = []
    for _, row in stations.iterrows():
        lat = float(row.get("Latitude", 0))
        lon = float(row.get("Longitude", 0))
        records.append({
            "station_id": _name_to_id(row["Station Name"]),
            "name": row["Station Name"].strip(),
            "latitude": lat,
            "longitude": lon,
            "elevation": float(row.get("Elevation", 0)),
            "network": "MN_NDAWN",
            "geometry": Point(lon, lat),
        })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    if bounds is not None:
        gdf = gdf.cx[bounds.west:bounds.east, bounds.south:bounds.north]

    logger.info("Found %d NDAWN stations", len(gdf))
    return gdf.reset_index(drop=True)


def fetch_hourly(
    station_ids: int | list[int],
    start: str,
    end: str,
    variable: str = "hdr",
) -> pd.DataFrame:
    """Fetch hourly observations from NDAWN for one or more stations.

    Parameters
    ----------
    station_ids:
        NDAWN station ID(s) (e.g. ``78`` or ``[78, 93, 220]``).
    start:
        Start date in ``"YYYY-MM-DD"`` format.
    end:
        End date in ``"YYYY-MM-DD"`` format (inclusive).
    variable:
        NDAWN variable code. Defaults to ``"hdr"`` (all hourly variables).

    Returns
    -------
    pandas.DataFrame
        Index: ``valid`` (UTC-ish datetime, NDAWN reports in CST/CDT).
        Columns: ``station_name``, ``latitude``, ``longitude``, plus
        variable columns.
    """
    if isinstance(station_ids, int):
        station_ids = [station_ids]

    station_params = "&".join(f"station={sid}" for sid in station_ids)

    url = (
        f"{_BASE_URL}?"
        f"{station_params}"
        f"&variable={variable}"
        f"&ttype=hourly"
        f"&quick_pick="
        f"&begin_date={start}"
        f"&end_date={end}"
    )

    text = _download(url)

    if not text:
        logger.warning("No NDAWN data returned for stations %s", station_ids)
        return pd.DataFrame()

    df = _parse_csv(text)

    # build a proper datetime column
    df = _add_datetime_index(df)

    return df


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _parse_csv(text: str) -> pd.DataFrame:
    """Parse NDAWN's multi-header CSV format.

    NDAWN CSVs have:
    - Row 0-2: metadata/blank rows
    - Row 3: column names
    - Row 4: unit row
    - Row 5+: data

    We skip the first 3 rows and the unit row (row index 4 → row 1
    after skipping 3).
    """
    df = pd.read_csv(
        StringIO(text),
        skiprows=3,
        header=0,
        na_values=["", " "],
    )

    # drop the units row (first data row after header)
    if len(df) > 0 and df.iloc[0].astype(str).str.contains(
        r"deg|mph|inch|Lys|ft|%|F$", case=False, regex=True
    ).any():
        df = df.iloc[1:].reset_index(drop=True)

    # strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # coerce numeric columns
    for col in df.columns:
        if col not in ("Station Name",):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _add_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Build a datetime index from Year/Month/Day/Hour columns."""
    required = {"Year", "Month", "Day", "Hour"}
    if not required.issubset(df.columns):
        logger.warning("NDAWN CSV missing date columns: %s", required - set(df.columns))
        return df

    df["valid"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour"]].astype(int).rename(
            columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour"}
        )
    )
    df = df.drop(columns=["Year", "Month", "Day", "Hour"], errors="ignore")
    df = df.set_index("valid").sort_index()

    # lowercase and clean up column names
    rename = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_")
        rename[col] = clean
    df = df.rename(columns=rename)

    return df


def _name_to_id(name: str) -> int | str:
    """Reverse lookup: station name → NDAWN station ID."""
    name_clean = name.strip().lower()
    for sid, sname in MN_STATIONS.items():
        if sname.lower() == name_clean:
            return sid
    return name.strip()


# ---------------------------------------------------------------------------
# Download helper with retry
# ---------------------------------------------------------------------------


def _download(url: str) -> str:
    """Fetch text data from NDAWN with retry."""
    logger.debug("Downloading: %s", url)
    # NDAWN sometimes rejects requests without a browser-like User-Agent
    headers = {"User-Agent": "met-timeseries/1.0"}
    for attempt in range(_MAX_ATTEMPTS):
        try:
            req = Request(url, headers=headers)  # noqa: S310
            data = urlopen(req, timeout=120).read().decode("utf-8")  # noqa: S310
            if data and "error" not in data[:100].lower():
                return data
        except Exception as exc:
            logger.warning(
                "Attempt %d/%d failed for NDAWN: %s",
                attempt + 1, _MAX_ATTEMPTS, exc,
            )
        time.sleep(2 ** attempt)

    logger.error("All %d download attempts failed for NDAWN", _MAX_ATTEMPTS)
    return ""