"""
Minimal protocol used to describe data source fetch functions.

No abstract base class is used; source modules simply expose a plain function
matching the ``FetchFunction`` signature.
"""

from __future__ import annotations

from typing import Protocol

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def plot_bounds(*boxes: tuple[BoundingBox, str], figsize=(10, 8)):
    """Plot one or more BoundingBox instances on a simple lat/lon axes.

    Parameters
    ----------
    *boxes:
        Tuples of ``(BoundingBox, label)`` to draw.
    figsize:
        Figure size.

    Example
    -------
    >>> from met_timeseries.sources.base import CACHE_BOUNDS, BoundingBox
    >>> metzone = BoundingBox(west=-94.5, south=44.0, east=-93.0, north=45.5)
    >>> plot_bounds((CACHE_BOUNDS, "MN Cache Bounds"), (metzone, "Metzone 1"))
    """
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, (bb, label) in enumerate(boxes):
        color = colors[i % len(colors)]
        width = bb.east - bb.west
        height = bb.north - bb.south
        rect = patches.Rectangle(
            (bb.west, bb.south), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.15,
            label=label,
        )
        ax.add_patch(rect)
        # Label at center
        ax.text(
            bb.west + width / 2,
            bb.south + height / 2,
            label,
            ha="center", va="center",
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Bounding Boxes")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.autoscale()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()