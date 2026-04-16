"""
NARR data source.

Architecture
------------
The module mirrors the PRISM source layout:

**Fetch layer** – :func:`fetch_narr`:
    Public entry point.  Checks the yearly cache, identifies missing
    variables, calls :func:`download` for anything not yet cached, merges
    into the cache, and returns a spatially subsetted dataset.

**Download layer** – :func:`download`:
    Downloads one or more NARR variables for a full year via OPeNDAP,
    iterating over the variable list internally (analogous to how
    :func:`prism.download` iterates over PRISM variables for one date).

NARR provides 3-hourly fields on a ~32 km Lambert Conformal Conic grid
with 2-D ``lat``/``lon`` coordinate variables.  Spatial subsetting uses
a bounding-box mask over those 2-D coordinates.

No authentication is required.

OPeNDAP URL pattern:
    https://psl.noaa.gov/thredds/dodsC/Datasets/NARR/monolevel/{variable}.{year}.nc
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from met_timeseries.sources.base import CACHE_BOUNDS, BoundingBox

logger = logging.getLogger(__name__)

_THREDDS_URL = (
    "https://psl.noaa.gov/thredds/dodsC/Datasets/NARR/monolevel/{variable}.{year}.nc"
)

_CACHE_PREFIX = "NARR_"

#: NARR variable names as they appear in the OPeNDAP file stems.
AVAILABLE_VARIABLES: dict[str, str] = {
    "air.2m":   "2-m air temperature (K)",
    "uwnd.10m": "10-m u-wind (m/s)",
    "vwnd.10m": "10-m v-wind (m/s)",
    "apcp":     "Accumulated precipitation, 3-hr (kg/m²)",
    "dswrf":    "Downward shortwave radiation (W/m²)",
    "dlwrf":    "Downward longwave radiation (W/m²)",
    "tcdc":     "Total cloud cover (%)",
    "pevap":    "Potential evaporation (kg/m²)",
    "snod":     "Snow depth (m)",
    "weasd":    "Water equivalent of snow (kg/m²)",
}

#: Default variables fetched when none are specified.
DEFAULT_VARIABLES: list[str] = [
    "air.2m",
    "uwnd.10m",
    "vwnd.10m",
    "apcp",
    "dswrf",
    "dlwrf",
    "tcdc",
    "pevap"
]

#: Mapping from OPeNDAP file stem to the internal NetCDF variable name.
_VARNAME_MAP: dict[str, str] = {
    "air.2m":   "air",
    "uwnd.10m": "uwnd",
    "vwnd.10m": "vwnd",
    "apcp":     "apcp",
    "dswrf":    "dswrf",
    "dlwrf":    "dlwrf",
    "tcdc":     "tcdc",
    "pevap":    "pevap",
    "snod":     "snod",
    "weasd":    "weasd",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_narr(
    year: int,
    cache_dir: Path | str,
    bounds: BoundingBox | None = None,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Fetch NARR 3-hourly data for a full year.

    Checks the yearly cache, downloads any missing variables via
    :func:`download`, merges into the cache, and returns a spatially
    subsetted dataset.

    Parameters
    ----------
    year:
        Calendar year to fetch.
    cache_dir:
        Directory for caching yearly NetCDF files.  Created automatically
        if it does not exist.
    bounds:
        Spatial bounding box in EPSG:4326.  Defaults to ``CACHE_BOUNDS``.
    variables:
        NARR variable file-stem names (e.g. ``["tcdc", "air.2m"]``).
        Defaults to :data:`DEFAULT_VARIABLES`.

    Returns
    -------
    :class:`xarray.Dataset`
        Dataset with the requested variables, dimensions ``(time, y, x)``,
        containing 3-hourly time steps for the full year.
    """

    if variables is None:
        variables = DEFAULT_VARIABLES

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{_CACHE_PREFIX}{year}.nc"

    if cache_path.exists():
        ds = _read_from_cache(cache_path)
        missing = [v for v in variables if _internal_varname(v) not in ds.data_vars]
        if missing:
            logger.info("Cache %s missing variables %s; downloading.", cache_path, missing)
            ds_missing = download(year=year, variables=missing)
            ds = xr.merge([ds, ds_missing])
            _write_to_cache(ds, cache_path)
    else:
        ds = download(year=year, variables=variables)
        _write_to_cache(ds, cache_path)


    if bounds is None:
        bounds = CACHE_BOUNDS

    ds = _clip_dataset(ds, bounds)

    return ds


# ---------------------------------------------------------------------------
# Download layer
# ---------------------------------------------------------------------------


def download(
    year: int,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Download NARR variables for a full year via OPeNDAP.

    Each NARR variable lives in its own OPeNDAP file
    (``{variable}.{year}.nc``).  This function iterates over the
    requested variables, downloads each one, and merges them on the
    shared ``(time, y, x)`` grid.

    Parameters
    ----------
    year:
        Calendar year to download.
    variables:
        List of NARR variable file-stem names (e.g. ``["tcdc", "air.2m"]``).
        Defaults to :data:`DEFAULT_VARIABLES`.

    Returns
    -------
    :class:`xarray.Dataset`
        Merged dataset with all requested variables for the full year.

    Raises
    ------
    RuntimeError
        If any OPeNDAP request fails.
    """
    if variables is None:
        variables = DEFAULT_VARIABLES

    datasets: list[xr.Dataset] = []

    for variable in variables:
        url = _THREDDS_URL.format(variable=variable, year=year)
        logger.info("Downloading NARR %s for %d from %s", variable, year, url)

        try:
            ds = xr.open_dataset(url)
            internal_name = _internal_varname(variable)
            if internal_name in ds:
                ds_out = ds[[internal_name]].load()
            else:
                # Fallback: keep all non-coordinate data variables
                data_vars = [
                    v for v in ds.data_vars
                    if v not in ("lat", "lon", "lat_bnds", "lon_bnds")
                ]
                ds_out = ds[data_vars].load()
            ds.close()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download NARR {variable} for {year}: {exc}"
            ) from exc

        datasets.append(ds_out)

    if len(datasets) == 1:
        return datasets[0]

    return xr.merge(datasets)


# ---------------------------------------------------------------------------
# Variable name mapping
# ---------------------------------------------------------------------------


def _internal_varname(file_stem: str) -> str:
    """Return the internal NetCDF variable name for a NARR file stem.

    Falls back to the file stem itself if no mapping is defined.
    """
    return _VARNAME_MAP.get(file_stem, file_stem)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _write_to_cache(ds: xr.Dataset, cache_path: Path) -> Path:
    """Write a dataset to disk with compression."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        var: {"zlib": True, "complevel": 4, "shuffle": True}
        for var in ds.data_vars
    }
    ds.to_netcdf(cache_path, encoding=encoding)
    logger.info("Cached NARR data to %s", cache_path)
    return cache_path


def _read_from_cache(cache_path: Path) -> xr.Dataset:
    """Load a cached yearly NARR dataset."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    ds = xr.open_dataset(cache_path)
    return ds


# ---------------------------------------------------------------------------
# Spatial subsetting
# ---------------------------------------------------------------------------


def _clip_dataset(ds: xr.Dataset, bounds: BoundingBox) -> xr.Dataset:
    """Subset NARR data to a bounding box using 2-D lat/lon coordinates.

    NARR uses a Lambert Conformal Conic projection with 2-D ``lat`` and
    ``lon`` coordinate variables on ``(y, x)`` dimensions.  This function
    builds a boolean mask from the bounding box and slices the ``y``/``x``
    dimensions to the minimal enclosing rectangle, buffered by one cell.

    Parameters
    ----------
    ds:
        NARR dataset with 2-D ``lat`` and ``lon`` variables.
    bounds:
        Target bounding box in EPSG:4326.

    Returns
    -------
    :class:`xarray.Dataset`
        Spatially subsetted dataset.
    """
    lat, lon = _get_latlon(ds)
    if lat is None:
        logger.warning("No lat/lon found in NARR dataset; skipping spatial subset")
        return ds

    mask = (
        (lat >= bounds.south)
        & (lat <= bounds.north)
        & (lon >= bounds.west)
        & (lon <= bounds.east)
    )

    y_dim = lat.dims[0]
    x_dim = lat.dims[1]

    y_any = mask.any(dim=x_dim)
    x_any = mask.any(dim=y_dim)

    y_indices = np.where(y_any.values)[0]
    x_indices = np.where(x_any.values)[0]

    if len(y_indices) == 0 or len(x_indices) == 0:
        logger.warning("No NARR grid cells found within %s", bounds)
        return ds

    y_min = max(0, int(y_indices.min()) - 1)
    y_max = min(ds.sizes[y_dim] - 1, int(y_indices.max()) + 1)
    x_min = max(0, int(x_indices.min()) - 1)
    x_max = min(ds.sizes[x_dim] - 1, int(x_indices.max()) + 1)

    return ds.isel({y_dim: slice(y_min, y_max + 1), x_dim: slice(x_min, x_max + 1)})


def _get_latlon(ds: xr.Dataset):
    """Return (lat, lon) DataArrays from the dataset, or (None, None)."""
    for lat_name, lon_name in [("lat", "lon"), ("latitude", "longitude")]:
        if lat_name in ds.coords or lat_name in ds:
            return ds[lat_name], ds[lon_name]
    return None, None