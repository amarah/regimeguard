"""Backtests must not turn unavailable quotes into flat trading days."""
import pandas as pd


def test_missing_quotes_are_excluded_from_all_backtest_series(monkeypatch, tmp_path):
    import regimeguard.backtest as backtest

    index = pd.bdate_range("2026-01-05", periods=8)
    prices = pd.DataFrame({
        "AAPL": [100, 101, 102, None, 104, 105, 106, 107],
        "SPY": [200, 201, 202, 203, 204, 205, 206, 207],
    }, index=index)
    monkeypatch.setattr(backtest, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(backtest, "get_prices", lambda *args, **kwargs: prices)
    monkeypatch.setattr(
        backtest,
        "_fit_regime_series",
        lambda _returns: pd.Series(1.0, index=index),
    )

    result = backtest.run_backtest(
        ["AAPL"], hedge_drag_annual=0, use_cached=False
    )

    curve = result["curve"]
    assert index[3] not in curve.index
    assert index[4] not in curve.index
    assert curve.index.equals(index[[1, 2, 5, 6, 7]])
    assert curve.notna().all().all()
