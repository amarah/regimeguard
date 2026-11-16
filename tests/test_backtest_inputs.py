"""Backtest inputs and cache keys should be safe and unambiguous."""
from unittest.mock import Mock

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
