"""Position sizing should reject inputs that cannot produce valid allocations."""
from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard.sizing import kelly_fraction


@pytest.mark.parametrize(
    "win_rate", [-0.01, 1.01, float("nan"), float("inf"), True]
)
def test_kelly_rejects_invalid_win_rate(win_rate):
    with pytest.raises(ValueError, match="Win rate"):
        kelly_fraction(win_rate, 2.0)


@pytest.mark.parametrize("payoff", [0, -1, float("nan"), float("inf"), "2"])
def test_kelly_rejects_invalid_payoff(payoff):
    with pytest.raises(ValueError, match="Payoff"):
        kelly_fraction(0.6, payoff)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan"), True])
def test_kelly_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="Kelly fraction"):
        kelly_fraction(0.6, 2.0, fraction)


@pytest.mark.parametrize("capital", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_capital_is_rejected_before_download(monkeypatch, capital):
    import regimeguard.sizing as sizing

    download = Mock()
    monkeypatch.setattr(sizing, "get_prices", download)

    with pytest.raises(ValueError, match="Capital must be a positive finite number"):
        sizing.size_positions(["AAPL"], capital, 1.0)

    download.assert_not_called()


@pytest.mark.parametrize("multiplier", [-0.1, float("nan"), float("inf"), "bad"])
def test_invalid_regime_multiplier_is_rejected_before_download(monkeypatch, multiplier):
    import regimeguard.sizing as sizing

    download = Mock()
    monkeypatch.setattr(sizing, "get_prices", download)

    with pytest.raises(ValueError, match="Regime multiplier must be a nonnegative finite number"):
        sizing.size_positions(["AAPL"], 10_000, multiplier)

    download.assert_not_called()


def test_direct_api_normalizes_and_deduplicates_tickers(monkeypatch):
    import regimeguard.sizing as sizing

    prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]})
    download = Mock(return_value=prices)
    monkeypatch.setattr(sizing, "get_prices", download)

    result = sizing.size_positions([" aapl ", "AAPL"], 10_000, 1.0)

    download.assert_called_once_with(["AAPL"], period="2y")
    assert result["ticker"].tolist() == ["AAPL"]


def test_missing_quotes_do_not_create_multi_day_daily_returns(monkeypatch):
    import regimeguard.sizing as sizing

    index = pd.bdate_range("2026-01-05", periods=6)
    prices = pd.DataFrame({
        "AAPL": [100.0, 101.0, None, 110.0, 111.0, 112.0],
    }, index=index)
    captured = {}

    def trade_stats(returns, lookback=126):
        captured["returns"] = returns
        return 0.5, 1.0

    monkeypatch.setattr(sizing, "get_prices", lambda *args, **kwargs: prices)
    monkeypatch.setattr(sizing, "estimate_trade_stats", trade_stats)

    sizing.size_positions(["AAPL"], 10_000, 1.0)

    expected = prices["AAPL"].pct_change(fill_method=None).dropna()
    pd.testing.assert_series_equal(captured["returns"], expected)
    assert index[2] not in captured["returns"].index
    assert index[3] not in captured["returns"].index


@pytest.mark.parametrize(
    "prices",
    [
        pd.DataFrame({"AAPL": [100.0]}),
        pd.DataFrame({"AAPL": [100.0, 101.0]}),
        pd.DataFrame({"AAPL": [100.0, None, 101.0]}),
    ],
)
def test_tickers_without_enough_returns_are_skipped(monkeypatch, prices):
    import regimeguard.sizing as sizing

    monkeypatch.setattr(sizing, "get_prices", lambda *args, **kwargs: prices)

    result = sizing.size_positions(["AAPL"], 10_000, 1.0)

    assert result.empty
