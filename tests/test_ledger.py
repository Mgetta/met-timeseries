"""Tests for the Ledger class."""

from __future__ import annotations

from pathlib import Path

import pytest

from met_timeseries.ledger import Ledger


def test_ledger_creates_file(tmp_output_dir: Path):
    ledger = Ledger(tmp_output_dir)
    ledger.mark_complete("nldas", 2020, 1)
    assert ledger.path.exists()


def test_mark_complete_and_is_complete(tmp_output_dir: Path):
    ledger = Ledger(tmp_output_dir)
    assert not ledger.is_complete("nldas", 2020, 1)
    ledger.mark_complete("nldas", 2020, 1)
    assert ledger.is_complete("nldas", 2020, 1)
    assert not ledger.is_complete("nldas", 2020, 2)
    assert not ledger.is_complete("prism", 2020, 1)


def test_mark_complete_idempotent(tmp_output_dir: Path):
    ledger = Ledger(tmp_output_dir)
    ledger.mark_complete("nldas", 2020, 1)
    ledger.mark_complete("nldas", 2020, 1)
    df = ledger.load()
    nldas_rows = df[(df["step"] == "nldas") & (df["year"] == 2020) & (df["month"] == 1)]
    assert len(nldas_rows) == 1


def test_load_empty(tmp_output_dir: Path):
    ledger = Ledger(tmp_output_dir)
    df = ledger.load()
    assert list(df.columns) == Ledger.COLUMNS
    assert len(df) == 0


def test_get_incomplete(tmp_output_dir: Path):
    ledger = Ledger(tmp_output_dir)
    ledger.mark_complete("nldas", 2020, 1)
    ledger.mark_complete("nldas", 2020, 3)
    incomplete = ledger.get_incomplete("nldas", 2020, 2020)
    # 12 months minus the 2 completed = 10 incomplete
    assert len(incomplete) == 10
    assert (2020, 2) in incomplete
    assert (2020, 1) not in incomplete
    assert (2020, 3) not in incomplete


def test_ledger_persists_across_instances(tmp_output_dir: Path):
    ledger1 = Ledger(tmp_output_dir)
    ledger1.mark_complete("prism", 2021, 6)

    ledger2 = Ledger(tmp_output_dir)
    assert ledger2.is_complete("prism", 2021, 6)
