"""
Minimal protocol used to describe data source fetch functions.

No abstract base class is used; source modules simply expose a plain function
matching the ``FetchFunction`` signature.
"""

from __future__ import annotations

from typing import Protocol

import xarray as xr


class BoundingBox:
    """Axis-aligned bounding box in geographic coordinates (EPSG:4326).

    Attributes
    ----------
    west, east:
        Longitude bounds (degrees).
    south, north:
        Latitude bounds (degrees).
    """

    __slots__ = ("west", "south", "east", "north")

    def __init__(self, west: float, south: float, east: float, north: float) -> None:
        self.west = west
        self.south = south
        self.east = east
        self.north = north

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BoundingBox(west={self.west}, south={self.south}, "
            f"east={self.east}, north={self.north})"
        )


class FetchFunction(Protocol):
    """Protocol satisfied by any ``fetch_*`` function in the sources package."""

    def __call__(
        self,
        bounds: BoundingBox,
        start: str,
        end: str | None = None,
        **kwargs: object,
    ) -> xr.Dataset: ...
