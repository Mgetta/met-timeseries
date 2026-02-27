"""
I/O helpers: save and load metzone timeseries Parquet files.

Output files are organised as::

    <output_dir>/<step>/<metzone_id>/<variable>/<year>_<month:02d>.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_timeseries(
    df: pd.DataFrame,
    output_dir: str,
    metzone_id: str | int,
    step: str,
    variable: str,
    year: int,
    month: int,
) -> Path:
    """Save a timeseries DataFrame to a Parquet file.

    The file is written to::

        <output_dir>/<step>/<metzone_id>/<variable>/<year>_<month:02d>.parquet

    A ``metzone_id`` column is added to *df* if not already present.

    Parameters
    ----------
    df:
        DataFrame containing the timeseries data to save.
    output_dir:
        Root output directory.
    metzone_id:
        Identifier for the metzone (used in both the directory path and a
        ``metzone_id`` column in the saved file).
    step:
        Pipeline step name (e.g. ``"nldas"`` or ``"prism"``).
    variable:
        Variable name (e.g. ``"precip"``).
    year:
        Calendar year.
    month:
        Calendar month (1–12).

    Returns
    -------
    pathlib.Path
        Path of the written Parquet file.
    """
    dest = (
        Path(output_dir)
        / step
        / str(metzone_id)
        / variable
        / f"{year}_{month:02d}.parquet"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    if "metzone_id" not in out.columns:
        out.insert(0, "metzone_id", metzone_id)

    out.to_parquet(dest, index=False)
    logger.debug("Saved %s", dest)
    return dest


def load_timeseries(path: str | Path) -> pd.DataFrame:
    """Load a timeseries Parquet file written by :func:`save_timeseries`.

    Parameters
    ----------
    path:
        Path to the Parquet file.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_parquet(path)
