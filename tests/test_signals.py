"""Tests for signal processing filters."""
from __future__ import annotations

from datetime import datetime
import pytz

import pytest
from signals.filters import MarketHoursFilter, SymbolNormalizer

IST = pytz.timezone("Asia/Kolkata")


class TestMarketHoursFilter:
    def _dt(self, weekday: int, hour: int, minute: int):
        """weekday: 0=Mon, 4=Fri, 5=Sat, 6=Sun"""
        # Build a date with the given weekday in 2025 (no major holidays in most weeks)
        from datetime import date, timedelta
        # 2025-01-06 is a Monday
        base = date(2025, 1, 6)
        target = base + timedelta(days=weekday)
        return IST.localize(datetime(target.year, target.month, target.day, hour, minute, 0))

    def test_open_during_market_hours(self):
        f = MarketHoursFilter()
        assert f.is_open(self._dt(0, 10, 0)) is True   # Mon 10:00 IST
        assert f.is_open(self._dt(4, 15, 0)) is True   # Fri 15:00 IST

    def test_closed_before_open(self):
        f = MarketHoursFilter()
        assert f.is_open(self._dt(1, 9, 0)) is False   # Tue 09:00 (before 09:15)

    def test_closed_after_close(self):
        f = MarketHoursFilter()
        assert f.is_open(self._dt(2, 15, 31)) is False  # Wed 15:31 (after 15:30)

    def test_closed_on_weekend(self):
        f = MarketHoursFilter()
        assert f.is_open(self._dt(5, 10, 0)) is False   # Sat 10:00
        assert f.is_open(self._dt(6, 10, 0)) is False   # Sun 10:00


class TestSymbolNormalizer:
    def test_strip_nse_prefix(self):
        n = SymbolNormalizer()
        assert n.normalize("NSE:RELIANCE") == "RELIANCE"
        assert n.normalize("BSE:500325") == "500325"

    def test_plain_symbol(self):
        n = SymbolNormalizer()
        assert n.normalize("HDFCBANK") == "HDFCBANK"

    def test_tradingview_alias(self):
        n = SymbolNormalizer()
        assert n.normalize("NIFTY50") == "NIFTY 50"
        assert n.normalize("NIFTYBANK") == "NIFTY BANK"

    def test_lowercase_input(self):
        n = SymbolNormalizer()
        assert n.normalize("nse:reliance") == "RELIANCE"
