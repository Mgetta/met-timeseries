import xarray as xr
from shapely.geometry import box


class BoundingBox:
    """Axis-aligned bounding box in geographic coordinates (EPSG:4326)."""

    __slots__ = ("west", "south", "east", "north")

    def __init__(self, west: float, south: float, east: float, north: float) -> None:
        self.west = west
        self.south = south
        self.east = east
        self.north = north

    def __repr__(self) -> str:
        return (
            f"BoundingBox(west={self.west}, south={self.south}, "
            f"east={self.east}, north={self.north})"
        )

    def contains(self, other: "BoundingBox") -> bool:
        """Return True if *other* is entirely within this box."""
        return (
            self.west <= other.west
            and self.south <= other.south
            and self.east >= other.east
            and self.north >= other.north
        )

#: Fixed clip boundary encompassing all HUC-8 watersheds that drain to or
#: from Minnesota.  Used as the download/cache extent for both PRISM and
#: NLDAS so that cached files are reusable across any metzone within MN.
#:
#: Approximate extent (rounded outward to the nearest 0.5°):
#:   West:  -97.5   (western MN border + Red River basin into ND/SD)
#:   South:  43.0   (southern MN border + Iowa/SD headwaters)
#:   East:  -89.0   (eastern MN border + St. Croix into WI)
#:   North:  50.0   (northern MN border + Rainy River into Ontario)
# This is more of a config and will eventually move there I suspect.
CACHE_BOUNDS = BoundingBox(west=-97.5, south=43.0, east=-89.0, north=50.0)

def bounds_to_polygon(bounds: BoundingBox):
    """Return a Shapely box polygon for *bounds*."""
    return box(bounds.west, bounds.south, bounds.east, bounds.north)


def clip_dataset(
    ds: xr.Dataset, 
    bounds: BoundingBox, 
    lat_dim: str = "lat", 
    lon_dim: str = "lon"
) -> xr.Dataset:
    """
    Clip an xarray Dataset to a spatial bounding box with automatic edge padding.
    Safely handles both ascending and descending latitude coordinates.
    """
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values

    # Pad by half a cell so we include cells whose edges overlap the bounds
    half_dy = abs(float(lats[1] - lats[0])) / 2
    half_dx = abs(float(lons[1] - lons[0])) / 2

    # Check if latitudes are descending (e.g., NLDAS) or ascending (e.g., PRISM)
    if lats[0] > lats[-1]:
        lat_slice = slice(bounds.north + half_dy, bounds.south - half_dy)
    else:
        lat_slice = slice(bounds.south - half_dy, bounds.north + half_dy)

    # Slice the dataset using a dictionary to support dynamic dimension names
    ds_clipped = ds.sel({
        lat_dim: lat_slice,
        lon_dim: slice(bounds.west - half_dx, bounds.east + half_dx),
    })
    
    return ds_clipped


