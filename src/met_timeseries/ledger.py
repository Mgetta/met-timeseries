"""Resumable processing tracker backed by a CSV ledger file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class Ledger:
    """CSV-backed ledger for tracking completed processing steps.

    Each row in the ledger represents a (step, year, month) triple that has
    been successfully processed.  The ledger is written atomically after each
    ``mark_complete`` call so that partial runs can be resumed safely.

    Parameters
    ----------
    output_dir : Path
        Root output directory; the ledger file is stored at
        ``{output_dir}/ledger.csv``.
    """

    COLUMNS = ["step", "year", "month", "completed_at"]

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "ledger.csv"
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._df is None:
            self._df = self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load the ledger from disk, creating an empty one if absent.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``step``, ``year``, ``month``,
            ``completed_at``.
        """
        if self.path.exists():
            df = pd.read_csv(self.path, dtype={"step": str, "year": int, "month": int})
        else:
            df = pd.DataFrame(columns=self.COLUMNS)
        self._df = df
        return df

    def mark_complete(self, step: str, year: int, month: int) -> None:
        """Record a (step, year, month) triple as successfully completed.

        Parameters
        ----------
        step : str
            Name of the processing step (e.g. ``"nldas"``, ``"prism"``).
        year : int
            Four-digit year.
        month : int
            Month number (1-12).
        """
        self._ensure_loaded()
        assert self._df is not None

        # Remove existing entry if present (idempotent)
        mask = (
            (self._df["step"] == step)
            & (self._df["year"] == year)
            & (self._df["month"] == month)
        )
        self._df = self._df[~mask].copy()

        new_row = pd.DataFrame(
            [
                {
                    "step": step,
                    "year": year,
                    "month": month,
                    "completed_at": pd.Timestamp.now("UTC").isoformat(),
                }
            ]
        )
        self._df = pd.concat([self._df, new_row], ignore_index=True)
        self._write()

    def is_complete(self, step: str, year: int, month: int) -> bool:
        """Check whether a (step, year, month) triple is marked complete.

        Parameters
        ----------
        step : str
            Name of the processing step.
        year : int
            Four-digit year.
        month : int
            Month number (1-12).

        Returns
        -------
        bool
            ``True`` if the entry exists in the ledger, ``False`` otherwise.
        """
        self._ensure_loaded()
        assert self._df is not None
        mask = (
            (self._df["step"] == step)
            & (self._df["year"] == year)
            & (self._df["month"] == month)
        )
        return bool(mask.any())

    def get_incomplete(
        self, step: str, start_year: int, end_year: int
    ) -> list[tuple[int, int]]:
        """Return all (year, month) pairs that have not yet been completed.

        Parameters
        ----------
        step : str
            Name of the processing step.
        start_year : int
            First year of the date range (inclusive).
        end_year : int
            Last year of the date range (inclusive).

        Returns
        -------
        list[tuple[int, int]]
            Sorted list of ``(year, month)`` tuples that are *not* yet
            marked complete in the ledger.
        """
        all_periods = [
            (y, m) for y in range(start_year, end_year + 1) for m in range(1, 13)
        ]
        return [(y, m) for y, m in all_periods if not self.is_complete(step, y, m)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write(self) -> None:
        """Persist the in-memory ledger to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        assert self._df is not None
        self._df.to_csv(self.path, index=False)
