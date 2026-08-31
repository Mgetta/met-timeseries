

import functools
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pyproj import Transformer
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Equal-area reprojection (module-level, created once on import)
# ---------------------------------------------------------------------------

_TO_EQUAL_AREA = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)


def _to_ea(geom: BaseGeometry) -> BaseGeometry:
    """Reproject a geometry from EPSG:4326 to EPSG:6933 (equal-area)."""
    return shapely_transform(_TO_EQUAL_AREA.transform, geom)


# ---------------------------------------------------------------------------
# Core weight computation
# ---------------------------------------------------------------------------
def _calculate_raster_coverage(
    polygon: BaseGeometry,
    lats: tuple[float, ...],
    lons: tuple[float, ...],
) -> float:
    """
    Calculates the exact fraction of a polygon's area that falls 
    within the bounding box of the raster grid.
    """
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)

    if len(lats_arr) < 2 or len(lons_arr) < 2:
        raise ValueError("At least 2 lat/lon coordinates are required.")

    # 1. Infer cell dimensions
    dx = abs(float(lons_arr[1] - lons_arr[0]))
    dy = abs(float(lats_arr[1] - lats_arr[0]))

    # 2. Create a single bounding box for the ENTIRE raster extent
    # We add half a cell width/height to the min/max coordinates to get the true edges
    raster_box = shapely_box(
        lons_arr.min() - dx / 2,
        lats_arr.min() - dy / 2,
        lons_arr.max() + dx / 2,
        lats_arr.max() + dy / 2,
    )

    # 3. Reproject both to equal-area to get true physical areas
    polygon_ea = _to_ea(polygon)
    raster_box_ea = _to_ea(raster_box)

    true_area = polygon_ea.area
    if true_area == 0:
        return 0.0

    # 4. Calculate the coverage fraction
    intersection = polygon_ea.intersection(raster_box_ea)
    return intersection.area / true_area

@functools.lru_cache(maxsize=128)
def _compute_weights_cached(
    polygon_wkb: bytes,
    lats: tuple[float, ...],
    lons: tuple[float, ...],
    normalize: bool
) -> np.ndarray:
    """LRU-cached implementation; keyed on hashable WKB + coordinate tuples."""
    from shapely import from_wkb

    polygon: BaseGeometry = from_wkb(polygon_wkb)

    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)

    if len(lats_arr) < 2:
        raise ValueError(
            "Cannot infer cell height from a single latitude value; "
            "at least 2 latitude coordinates are required."
        )
    if len(lons_arr) < 2:
        raise ValueError(
            "Cannot infer cell width from a single longitude value; "
            "at least 2 longitude coordinates are required."
        )

    dx = abs(float(lons_arr[1] - lons_arr[0]))
    dy = abs(float(lats_arr[1] - lats_arr[0]))

    weights = np.zeros((len(lats_arr), len(lons_arr)), dtype=float)

    # 1. Get the polygon's bounding box
    minx, miny, maxx, maxy = polygon.bounds
    
    # 2. Add a buffer of 1 cell width/height to ensure we catch edge overlaps
    minx -= dx
    maxx += dx
    miny -= dy
    maxy += dy

    for i, lat in enumerate(lats_arr):
        # SKIP if the latitude row is entirely outside the polygon's bounding box
        if not (miny <= lat <= maxy):
            continue
            
        for j, lon in enumerate(lons_arr):
            # SKIP if the longitude column is entirely outside the polygon's bounding box
            if not (minx <= lon <= maxx):
                continue
                
            # Only do the expensive math if the cell is actually near the polygon
            cell = shapely_box(
                lon - dx / 2, lat - dy / 2, lon + dx / 2, lat + dy / 2
            )
            intersection = cell.intersection(polygon)
            if intersection.is_empty:
                continue

            cell_ea = _to_ea(cell)
            inter_ea = _to_ea(intersection)
            weights[i, j] = inter_ea.area / cell_ea.area


    if normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total

    return weights


def compute_weights(
    polygon: BaseGeometry,
    lats: tuple[float, ...],
    lons: tuple[float, ...],
    *,
    normalize: bool = False
) -> np.ndarray:
    """Return a 2-D weight array (shape ``len(lats) × len(lons)``).

    Each weight represents the fractional area overlap between a grid cell
    and the polygon, computed in an equal-area projection (EPSG:6933).

    Results are LRU-cached on the polygon's WKB representation and the
    coordinate tuples, so repeated calls with the same inputs are free.

    Parameters
    ----------
    polygon:
        Shapely geometry in EPSG:4326.
    lats:
        Grid cell center latitudes (tuple of floats).
    lons:
        Grid cell center longitudes (tuple of floats).
    normalize:
        If ``False`` (default), each weight is ``intersection_area /
        cell_area`` — the fraction of the *cell* covered by the polygon.
        The caller must divide by ``weights.sum()`` at application time.

        If ``True``, weights are rescaled so they sum to 1.0.  The result
        can be used directly as ``(data * weights).sum()`` without further
        normalization.

    Returns
    -------
    numpy.ndarray
        2-D float array aligned to ``(lat, lon)``.

    Raises
    ------
    ValueError
        If *lats* or *lons* contain fewer than 2 values (cell spacing
        cannot be inferred from a single coordinate).
    """
    return _compute_weights_cached(polygon.wkb, lats, lons, normalize)

def build_weightmap(
    datarray: xr.Dataset,
    polygons: gpd.GeoDataFrame,
    poly_id_columns: str | list[str] | None = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    min_coverage: float = 0.99,
) -> xr.Dataset:
    """Core engine for computing spatial weights across one or many polygons."""
    
    lats = datarray[lat_dim].values
    lons = datarray[lon_dim].values
    lats_tuple = tuple(float(v) for v in lats)
    lons_tuple = tuple(float(v) for v in lons)

    # 1. PRE-ALLOCATE THE 3D ARRAY
    weights_3d = np.zeros((len(polygons), len(lats), len(lons)), dtype=datarray[lat_dim].dtype)

   

    # 2. Sequential processing
    for i, (_, row) in enumerate(polygons.iterrows()):
        geom = row.geometry
        
        # Fast coverage check (using the function we built earlier)
        coverage = _calculate_raster_coverage(geom, lats_tuple, lons_tuple)
        
        if coverage < min_coverage:
            continue #TODO: log a warning here, but don't raise an error — we can still compute weights for partial coverage, we just want to be aware of it.
            # raise ValueError(
            #     f"Polygon at index {i} has insufficient coverage ({coverage:.2%}) of the raster grid. "
            #     "Consider increasing 'min_coverage' or checking the polygon's geometry and the raster extent."
            # ) 

        # Heavy weight computation
        w = compute_weights(geom, lats_tuple, lons_tuple, normalize=False)
        total = w.sum()
        
        if total == 0:
            raise ValueError(f"Polygon {i} has 0 total weight despite passing coverage.")
            
        weights_3d[i, :, :] = (w / total)


    valid_mask = datarray.isel(time=0).load().notnull().values  # shape (lat, lon)
    weights_3d = weights_3d * valid_mask[np.newaxis, :, :]
    
    # 3. Build coordinates
    coords = {
        "polygon_index": np.arange(len(polygons)),
        lat_dim: lats,
        lon_dim: lons
    }
    
    # Safely handle ID columns if they were provided
    if poly_id_columns:
        for col in poly_id_columns:
            coords[col] = ("polygon_index", polygons[col].to_numpy(dtype=object))
    
    
    return xr.Dataset(
        data_vars={"weights": (["polygon_index", lat_dim, lon_dim], weights_3d)},
        coords=coords,
        attrs={"description": "Precomputed spatial weights for polygon aggregation"},
    )

# ---------------------------------------------------------------------------
# High-level aggregation functions
# ---------------------------------------------------------------------------


def get_weights(
    polygon: BaseGeometry,
    dataset: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    min_coverage: float = 0.99,
) -> xr.Dataset:
    """
    Return grid-cell weights for a single polygon as an xarray Dataset.
    This wraps build_weightmap to guarantee identical output formats.
    """
    # Wrap the single geometry in a temporary GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
    
    # Call the engine; if the polygon is out of bounds, the engine 
    # will automatically raise the ValueError for us.
    return build_weightmap(
        dataset=dataset,
        polygons=gdf,
        poly_id_columns=None,
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        min_coverage=min_coverage
    )

def weighted_mean_timeseries(
    dataset: xr.Dataset,
    polygon: BaseGeometry,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    """Collapse a (time, lat, lon) Dataset to one (time,) Dataset."""
    w_da = get_weights(polygon, dataset, lat_dim=lat_dim, lon_dim=lon_dim)["weights"]
    # Native xarray multiplication and sum over spatial dimensions!
    # This automatically handles 2D, 3D, and NaN values, keeping time coordinates intact.
    result_ds = (dataset * w_da['weights']).sum(dim=[lat_dim, lon_dim])
    return result_ds

def _weight_dataset(
    dataset: xr.Dataset,
    weights: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.Dataset:
    """Collapse a (time, lat, lon) Dataset to one (time,) Dataset."""
    # Native xarray multiplication and sum over spatial dimensions!
    # This automatically handles 2D, 3D, and NaN values, keeping time coordinates intact.
    result_ds = (dataset * weights['weights']).sum(dim=[lat_dim, lon_dim])
    return result_ds


def save_weightmap(weight_ds: xr.Dataset, path: str | Path):
    encoding = {
        "weights": {
            "zlib": True,       # Turn on compression
            "complevel": 5,     # 1 is fastest, 9 is smallest. 4-5 is the sweet spot.
            # Optional but highly recommended: chunking
            # This makes reading individual polygons much faster
            "chunksizes": (1, len(weight_ds.lat), len(weight_ds.lon)) 
        }
    }
    weight_ds.to_netcdf(path, engine="netcdf4", encoding=encoding)



def plot_weights(
    polygon: BaseGeometry,
    weights: xr.Dataset,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> "matplotlib.axes.Axes":
    """Plot the weight grid for a polygon overlaid on the dataset grid."""
    import matplotlib.pyplot as plt


    #weights = weights["weights"].where(weights["weights"] > 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("lightgrey")

    weights.plot()
    # weights.plot.pcolormesh(
    #     x=lon_dim, y=lat_dim,
    #     ax=ax,
    #     cmap="YlOrRd",
    #     edgecolors="grey",
    #     linewidths=0.3,
    #     cbar_kwargs={"label": "Weight", "shrink": 0.8},
    # )

    gpd.GeoSeries([polygon], crs="EPSG:4326").boundary.plot(
        ax=ax, color="blue", linewidth=2, label="Polygon of interest",
    )

    # extent = union of grid bounds + polygon bounds, with padding
    lats = weights[lat_dim].values
    lons = weights[lon_dim].values
    dy = abs(float(lats[1] - lats[0])) / 2
    dx = abs(float(lons[1] - lons[0])) / 2
    poly_bounds = polygon.bounds  # (minx, miny, maxx, maxy)

    xmin =  poly_bounds[0] - dx*2
    xmax = poly_bounds[2] + dx*2
    ymin = poly_bounds[1] - dy*2
    ymax = poly_bounds[3] + dy*2

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.set_title("Grid-Cell Weights")

    return ax