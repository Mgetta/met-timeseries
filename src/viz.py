
from met_timeseries.geometry import BoundingBox
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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