"""Stress windows must use actual aligned observations, not filled prices."""
import pandas as pd

from regimeguard import stress


def test_missing_prices_do_not_create_synthetic_stress_returns(monkeypatch):
    index = pd.date_range("2026-01-01", periods=6)
    prices = pd.DataFrame({
        "A": [100.0, 110.0, None, 80.0, 100.0, 120.0],
        "SPY": [100.0, 110.0, 80.0, 80.0, 100.0, 120.0],
    }, index=index)
    requested = []

    def get_prices(tickers, period):
        requested.append(tickers)
        return prices

    monkeypatch.setattr(stress, "get_prices", get_prices)
    monkeypatch.setattr(stress, "STRESS_EVENTS", {
        "gap": (str(index[0].date()), str(index[-1].date()))})

    result = stress.run_stress_tests({"A": 1.0})

    assert requested == [["A", "SPY"]]
    # Only the observed adjacent returns remain: +10%, +25%, and +20%.
    assert result.loc[0, "portfolio_return"] == 0.65


def test_spy_is_requested_once_when_it_is_a_holding(monkeypatch):
    index = pd.date_range("2026-01-01", periods=6)
    prices = pd.DataFrame({"SPY": [100, 101, 102, 103, 104, 105]}, index=index)
    requested = []
    monkeypatch.setattr(stress, "get_prices", lambda tickers, period: requested.append(tickers) or prices)
    monkeypatch.setattr(stress, "STRESS_EVENTS", {
        "test": (str(index[0].date()), str(index[-1].date()))})

    stress.run_stress_tests({"SPY": 1.0})

    assert requested == [["SPY"]]


def test_window_without_aligned_returns_is_skipped(monkeypatch):
    index = pd.date_range("2026-01-01", periods=6)
    prices = pd.DataFrame({
        "A": [100.0, None, 101.0, None, 102.0, None],
        "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
    }, index=index)
    monkeypatch.setattr(stress, "get_prices", lambda *args, **kwargs: prices)
    monkeypatch.setattr(stress, "STRESS_EVENTS", {
        "sparse": (str(index[0].date()), str(index[-1].date()))})

    assert stress.run_stress_tests({"A": 1.0}).empty
