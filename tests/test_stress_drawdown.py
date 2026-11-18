"""Stress drawdown includes the portfolio's starting value."""
import pandas as pd

from regimeguard import stress


def test_first_observed_loss_counts_toward_drawdown(monkeypatch):
    index = pd.date_range("2026-01-01", periods=6)
    prices = pd.DataFrame({
        "A": [100.0, 80.0, 88.0, 96.8, 106.48, 117.128],
        "SPY": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    }, index=index)
    monkeypatch.setattr(stress, "get_prices", lambda *args, **kwargs: prices)
    monkeypatch.setattr(stress, "STRESS_EVENTS", {
        "initial loss": (str(index[0].date()), str(index[-1].date())),
    })

    result = stress.run_stress_tests({"A": 1.0})

    assert result.loc[0, "portfolio_max_dd"] == -0.2
