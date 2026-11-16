"""Flat observations should not bias estimated trade outcomes."""
import pandas as pd
import pytest

from regimeguard.sizing import estimate_trade_stats


def test_flat_returns_are_neutral():
    returns = pd.Series([0.10, 0.0, -0.10, 0.0])

    win_rate, payoff = estimate_trade_stats(returns)

    assert win_rate == pytest.approx(0.5)
    assert payoff == pytest.approx(1.0)


def test_all_flat_history_uses_neutral_fallback():
    win_rate, payoff = estimate_trade_stats(pd.Series([0.0, 0.0, 0.0]))

    assert win_rate == pytest.approx(0.5)
    assert payoff == pytest.approx(1.0)


def test_lookback_is_applied_before_classifying_outcomes():
    returns = pd.Series([-0.50, 0.10, -0.05, 0.0])

    win_rate, payoff = estimate_trade_stats(returns, lookback=3)

    assert win_rate == pytest.approx(0.5)
    assert payoff == pytest.approx(2.0)
