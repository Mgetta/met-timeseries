"""Partitioned Parquet read/write helpers for met-timeseries output."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_monthly_output(
    df: pd.DataFrame,
    output_dir: Path | str,
    step: str,
    year: int,
    month: int,
) -> Path:
    """Save a monthly DataFrame to a partitioned Parquet file.

    Writes the data to:
    ``{output_dir}/{step}/year={year}/month={month:02d}/data.parquet``

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    output_dir : Path or str
        Root output directory.
    step : str
        Processing step name (e.g. ``"nldas"``, ``"prism"``).
    year : int
        Four-digit year.
    month : int
        Month number (1-12).

    Returns
    -------
    Path
        Path to the written Parquet file.
    """
    output_dir = Path(output_dir)
    partition_dir = output_dir / step / f"year={year}" / f"month={month:02d}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    out_path = partition_dir / "data.parquet"
    df.to_parquet(out_path, index=True)
    logger.debug("Saved %d rows to %s", len(df), out_path)
    return out_path


def read_output(
    output_dir: Path | str,
    step: str,
    year_range: tuple[int, int] | None = None,
    polygon_ids: list | None = None,
) -> pd.DataFrame:
    """Read partitioned Parquet output with optional partition pruning.

    Reads data from:
    ``{output_dir}/{step}/year={year}/month={month:02d}/data.parquet``

    Parameters
    ----------
    output_dir : Path or str
        Root output directory.
    step : str
        Processing step name (e.g. ``"nldas"``, ``"prism"``).
    year_range : tuple[int, int] or None
        Inclusive ``(start_year, end_year)`` range to filter partitions.
        If ``None``, all available years are read.
    polygon_ids : list or None
        List of polygon IDs to filter rows.  Applied after reading each
        partition.  If ``None``, all rows are returned.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all matching partitions.

    Raises
    ------
    FileNotFoundError
        If the step directory does not exist.
    """
    output_dir = Path(output_dir)
    step_dir = output_dir / step
    if not step_dir.exists():
        raise FileNotFoundError(f"Step directory not found: {step_dir}")

    frames: list[pd.DataFrame] = []
    year_dirs = sorted(step_dir.glob("year=*"))

    for year_dir in year_dirs:
        year = int(year_dir.name.split("=")[1])
        if year_range is not None:
            if year < year_range[0] or year > year_range[1]:
                continue

        for month_dir in sorted(year_dir.glob("month=*")):
            parquet_path = month_dir / "data.parquet"
            if not parquet_path.exists():
                continue
            df = pd.read_parquet(parquet_path)
            if polygon_ids is not None:
                # Try to filter by index or polygon_id column
                if "polygon_id" in df.columns:
                    df = df[df["polygon_id"].isin(polygon_ids)]
                elif df.index.name == "polygon_id":
                    df = df[df.index.isin(polygon_ids)]
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=False)
