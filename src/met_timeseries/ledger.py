"""
Ledger: track which pipeline steps have been completed.

All state is stored in a CSV file with columns: ``step``, ``year``, ``month``.
Each row records one completed (step, year, month) combination.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_COLUMNS = ["step", "year", "month"]


def load_ledger(path: str) -> pd.DataFrame:
    """Load the ledger CSV, returning an empty DataFrame if it does not exist.

    Parameters
    ----------
    path:
        Path to the CSV ledger file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``step``, ``year``, ``month``.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(p, dtype={"step": str, "year": int, "month": int})
    # Ensure expected columns are present
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[_COLUMNS]


def mark_complete(path: str, step: str, year: int, month: int) -> None:
    """Append a completed (step, year, month) entry to the ledger.

    Parameters
    ----------
    path:
        Path to the CSV ledger file (created if it does not exist).
    step:
        Name of the pipeline step (e.g. ``"nldas"`` or ``"prism"``).
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    """
    df = load_ledger(path)
    new_row = pd.DataFrame([{"step": step, "year": int(year), "month": int(month)}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)
    logger.debug("Marked complete: step=%s year=%d month=%d", step, year, month)


def is_complete(path: str, step: str, year: int, month: int) -> bool:
    """Return ``True`` if (step, year, month) is recorded in the ledger.

    Parameters
    ----------
    path:
        Path to the CSV ledger file.
    step:
        Pipeline step name.
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    """
    df = load_ledger(path)
    if df.empty:
        return False
    mask = (df["step"] == step) & (df["year"] == int(year)) & (df["month"] == int(month))
    return bool(mask.any())


def get_incomplete(
    path: str,
    step: str,
    start_year: int,
    end_year: int,
) -> list[tuple[int, int]]:
    """Return a list of (year, month) tuples not yet recorded in the ledger.

    Parameters
    ----------
    path:
        Path to the CSV ledger file.
    step:
        Pipeline step name.
    start_year:
        First year to consider (inclusive).
    end_year:
        Last year to consider (inclusive).

    Returns
    -------
    list of (year, month) tuples
        All (year, month) pairs in the date range that are **not** yet marked
        complete for *step*.
    """
    result: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if not is_complete(path, step, year, month):
                result.append((year, month))
    return result
