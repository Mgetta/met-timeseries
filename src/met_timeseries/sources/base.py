"""
Minimal protocol used to describe data source fetch functions.

No abstract base class is used; source modules simply expose a plain function
matching the ``FetchFunction`` signature.
"""

from __future__ import annotations

from typing import Protocol

import xarray as xr


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
CACHE_BOUNDS = BoundingBox(west=-97.5, south=43.0, east=-89.0, north=50.0)


class FetchFunction(Protocol):
    """Protocol satisfied by any ``fetch_*`` function in the sources package."""

    def __call__(
        self,
        bounds: BoundingBox,
        start: str,
        end: str | None = None,
        **kwargs: object,
    ) -> xr.Dataset: ...
