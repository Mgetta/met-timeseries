"""
Tests for met_timeseries.ledger — module-level function API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


class TestLoadLedger:
    def test_returns_empty_df_when_missing(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import load_ledger

        df = load_ledger(str(ledger_path))
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert list(df.columns) == ["step", "year", "month"]

    def test_loads_existing_file(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import load_ledger, mark_complete

        mark_complete(str(ledger_path), "nldas", 2010, 1)
        df = load_ledger(str(ledger_path))
        assert len(df) == 1
        assert df.iloc[0]["step"] == "nldas"
        assert int(df.iloc[0]["year"]) == 2010
        assert int(df.iloc[0]["month"]) == 1


class TestMarkComplete:
    def test_creates_file_if_missing(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import mark_complete

        mark_complete(str(ledger_path), "nldas", 2000, 6)
        assert ledger_path.exists()

    def test_appends_entries(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import mark_complete, load_ledger

        mark_complete(str(ledger_path), "nldas", 2000, 1)
        mark_complete(str(ledger_path), "nldas", 2000, 2)
        mark_complete(str(ledger_path), "prism", 2000, 1)
        df = load_ledger(str(ledger_path))
        assert len(df) == 3

    def test_accepts_int_types(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import mark_complete, is_complete

        mark_complete(str(ledger_path), "nldas", 2005, 12)
        assert is_complete(str(ledger_path), "nldas", 2005, 12)


class TestIsComplete:
    def test_returns_false_when_not_recorded(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import is_complete

        assert not is_complete(str(ledger_path), "nldas", 2000, 1)

    def test_returns_true_after_mark(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import is_complete, mark_complete

        mark_complete(str(ledger_path), "nldas", 2000, 1)
        assert is_complete(str(ledger_path), "nldas", 2000, 1)

    def test_step_is_distinct(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import is_complete, mark_complete

        mark_complete(str(ledger_path), "nldas", 2000, 1)
        assert not is_complete(str(ledger_path), "prism", 2000, 1)

    def test_month_is_distinct(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import is_complete, mark_complete

        mark_complete(str(ledger_path), "nldas", 2000, 1)
        assert not is_complete(str(ledger_path), "nldas", 2000, 2)

    def test_year_is_distinct(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import is_complete, mark_complete

        mark_complete(str(ledger_path), "nldas", 2000, 1)
        assert not is_complete(str(ledger_path), "nldas", 2001, 1)


class TestGetIncomplete:
    def test_all_incomplete_when_ledger_empty(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import get_incomplete

        result = get_incomplete(str(ledger_path), "nldas", 2000, 2000)
        assert len(result) == 12  # 12 months

    def test_completed_months_excluded(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import get_incomplete, mark_complete

        for m in range(1, 13):
            mark_complete(str(ledger_path), "nldas", 2000, m)

        result = get_incomplete(str(ledger_path), "nldas", 2000, 2000)
        assert result == []

    def test_partial_completion(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import get_incomplete, mark_complete

        for m in range(1, 7):  # Jan–Jun complete
            mark_complete(str(ledger_path), "nldas", 2000, m)

        result = get_incomplete(str(ledger_path), "nldas", 2000, 2000)
        assert result == [(2000, m) for m in range(7, 13)]

    def test_multi_year_range(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import get_incomplete

        result = get_incomplete(str(ledger_path), "nldas", 2000, 2001)
        assert len(result) == 24  # 2 years × 12 months

    def test_step_not_mixed(self, ledger_path: Path) -> None:
        from met_timeseries.ledger import get_incomplete, mark_complete

        # Mark all nldas months complete
        for m in range(1, 13):
            mark_complete(str(ledger_path), "nldas", 2000, m)

        # prism should still be all incomplete
        result = get_incomplete(str(ledger_path), "prism", 2000, 2000)
        assert len(result) == 12
