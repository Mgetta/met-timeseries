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
import xesmf as xe

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
    regrid: bool = True,
    regrid_resolution: float = 0.3
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
            ds_cached = ds.load() # read into RAM
            ds.close() # close the file handle as we will overwrite the cache with the merged dataset
            ds = xr.merge([ds_cached, ds_missing])
            _write_to_cache(ds,cache_path)
    else:
        ds = download(year=year, variables=variables)
        _write_to_cache(ds, cache_path)

    
    if regrid:
        ds = _regrid(ds, CACHE_BOUNDS, regrid_resolution)


    if bounds is not None:
        ds = _clip_dataset(ds, bounds=bounds)
        

    return ds.load()


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

def _regrid(
    ds: xr.Dataset,
    bounds: BoundingBox,
    resolution: float,
) -> xr.Dataset:
    """Regrid NARR from Lambert Conformal (y, x) to rectilinear (lat, lon).

    Builds the target grid from *bounds* and *resolution*, constructs
    an xesmf bilinear regridder, and applies it.

    Parameters
    ----------
    ds:
        NARR dataset with 2-D ``lat(y, x)`` and ``lon(y, x)`` coordinates.
    bounds:
        Target bounding box in EPSG:4326.
    resolution:
        Target grid spacing in degrees.

    Returns
    -------
    :class:`xarray.Dataset`
        Dataset with 1-D ``lat`` and ``lon`` dimensions.
    """
    import xesmf as xe

    ds_target = xr.Dataset({
        "lat": (["lat"], np.arange(bounds.south, bounds.north, resolution)),
        "lon": (["lon"], np.arange(bounds.west, bounds.east, resolution)),
    })

    regridder = xe.Regridder(ds, ds_target, method="bilinear")
    return regridder(ds)


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
#: NARR native projection — Lambert Conformal Conic.
#:
#: Parameters from the NCEP NARR grid definition (Grid 221):
#:   - Standard parallels: 50°N (both — tangent case)
#:   - Central meridian: 107°W
#:   - Reference latitude: 50°N
#:   - False easting / northing: derived from the grid origin
#:   - Resolution: 32.463 km at the standard parallel
#:   - Datum: WGS84
_NARR_PROJ4 = (
    "+proj=lcc "
    "+lat_1=50 "
    "+lat_2=50 "
    "+lat_0=50 "
    "+lon_0=-107 "
    "+x_0=5632642.22547 "
    "+y_0=4612545.65137 "
    "+datum=WGS84 "
    "+units=m "
    "+no_defs"
)

def _clip_dataset(
    ds: xr.Dataset,
    bounds: BoundingBox,
) -> xr.Dataset:
    """Clip an xarray Dataset to the given spatial bounding box."""
    lats = ds.lat.values
    lons = ds.lon.values

    # pad by half a cell so we include cells whose edges overlap the bounds
    half_dy = abs(float(lats[1] - lats[0])) / 2
    half_dx = abs(float(lons[1] - lons[0])) / 2

    if lats[0] > lats[-1]:
        lat_slice = slice(bounds.north + half_dy, bounds.south - half_dy)
    else:
        lat_slice = slice(bounds.south - half_dy, bounds.north + half_dy)

    ds = ds.sel(
        lat=lat_slice,
        lon=slice(bounds.west - half_dx, bounds.east + half_dx),
    )
    return ds

def _clip_conformal_dataset(ds: xr.Dataset, bounds: BoundingBox) -> xr.Dataset:
    """Subset NARR data to a bounding box using rioxarray.

    Clips in the native Lambert Conformal projection space, which
    correctly handles grid rotation and cell overlap at the edges.
    """
    import rioxarray  # noqa: F401

    # Ensure CRS is set
    if ds.rio.crs is None:
        ds = ds.rio.write_crs(_NARR_PROJ4)

    # Set spatial dims if not already recognized
    y_dim = "y" if "y" in ds.dims else list(ds.dims)[-2]
    x_dim = "x" if "x" in ds.dims else list(ds.dims)[-1]
    ds = ds.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)

    # clip_box reprojects the bounds into the dataset's CRS internally
    ds = ds.rio.clip_box(
        minx=bounds.west,
        miny=bounds.south,
        maxx=bounds.east,
        maxy=bounds.north,
        crs="EPSG:4326",
    )

    return ds

def _get_latlon(ds: xr.Dataset):
    """Return (lat, lon) DataArrays from the dataset, or (None, None)."""
    for lat_name, lon_name in [("lat", "lon"), ("latitude", "longitude")]:
        if lat_name in ds.coords or lat_name in ds:
            return ds[lat_name], ds[lon_name]
    return None, None