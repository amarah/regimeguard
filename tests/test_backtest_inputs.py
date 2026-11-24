"""Backtest inputs and cache keys should be safe and unambiguous."""
from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.mark.parametrize("capital", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_capital_is_rejected_before_download(monkeypatch, capital):
    import regimeguard.backtest as backtest

    download = Mock()
    monkeypatch.setattr(backtest, "get_prices", download)

    with pytest.raises(ValueError, match="Capital must be a positive finite number"):
        backtest.run_backtest(["AAPL"], capital=capital)

    download.assert_not_called()


@pytest.mark.parametrize("drag", [-0.01, float("nan"), float("inf"), "bad"])
def test_invalid_hedge_drag_is_rejected_before_download(monkeypatch, drag):
    import regimeguard.backtest as backtest

    download = Mock()
    monkeypatch.setattr(backtest, "get_prices", download)

    with pytest.raises(ValueError, match="hedge drag must be a nonnegative finite number"):
        backtest.run_backtest(["AAPL"], hedge_drag_annual=drag)

    download.assert_not_called()


@pytest.mark.parametrize("period", ["", "   ", None, 10])
def test_invalid_period_is_rejected_before_download(monkeypatch, period):
    import regimeguard.backtest as backtest

    download = Mock()
    monkeypatch.setattr(backtest, "get_prices", download)

    with pytest.raises(ValueError, match="Period must be a nonempty string"):
        backtest.run_backtest(["AAPL"], period=period)

    download.assert_not_called()


def test_cache_identity_is_unambiguous_and_stays_in_cache_directory(monkeypatch, tmp_path):
    import regimeguard.backtest as backtest

    monkeypatch.setattr(backtest, "CACHE_DIR", str(tmp_path))
    first = backtest._backtest_cache_path(["A_B", "C"], "10y")
    second = backtest._backtest_cache_path(["A", "B_C"], "10y")
    hostile = backtest._backtest_cache_path(["AAPL"], "../../outside")

    assert first != second
    assert hostile.startswith(str(tmp_path))
    assert ".." not in hostile


def test_regime_cache_is_tied_to_the_price_history(tmp_path):
    import regimeguard.backtest as backtest

    path = str(tmp_path / "regime.pkl")
    multipliers = pd.Series([1.0, 0.6], index=pd.date_range("2026-01-01", periods=2))
    backtest._write_regime_cache(path, "first-history", multipliers)

    pd.testing.assert_series_equal(
        backtest._load_regime_cache(path, "first-history"), multipliers
    )
    assert backtest._load_regime_cache(path, "changed-history") is None


def test_corrupt_regime_cache_is_ignored(tmp_path):
    import regimeguard.backtest as backtest

    path = tmp_path / "regime.pkl"
    path.write_bytes(b"not a pickle")

    assert backtest._load_regime_cache(str(path), "history") is None
