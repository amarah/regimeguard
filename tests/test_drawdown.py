"""Regression tests for losses before the first observed equity peak."""
import numpy as np
import pandas as pd
import pytest


class ZeroShocks:
    def standard_normal(self, shape):
        return np.zeros(shape)


@pytest.mark.parametrize('daily_return, expected', [(-0.1, 0.1), (0.1, 0.0)])
def test_one_day_drawdown_includes_starting_capital(monkeypatch, daily_return, expected):
    import regimeguard.risk as risk
    prices = pd.DataFrame({'A': 100 * (1 + daily_return) ** np.arange(5)})
    monkeypatch.setattr(risk, 'get_prices', lambda *args, **kwargs: prices)
    monkeypatch.setattr(risk.np.random, 'default_rng', lambda *args: ZeroShocks())
    result = risk.monte_carlo_cvar({'A': 1.0}, horizon_days=1, sims=10, student_t=False)
    assert result['DaR_95'] == pytest.approx(expected)


@pytest.mark.parametrize('path, expected', [
    ([100, 90, 80, 85, 100], -0.2),
    ([100, 110, 88, 100, 120], -0.2),
    ([100, 110, 120, 130, 140], 0.0),
])
def test_stress_drawdown_uses_initial_and_later_peaks(monkeypatch, path, expected):
    import regimeguard.stress as stress
    dates = pd.date_range('2026-01-01', periods=5)
    prices = pd.DataFrame({'A': path, 'SPY': path}, index=dates)
    monkeypatch.setattr(stress, 'get_prices', lambda *args, **kwargs: prices)
    monkeypatch.setattr(stress, 'STRESS_EVENTS', {'example': ('2026-01-01', '2026-01-05')})
    report = stress.run_stress_tests({'A': 1.0})
    assert report.iloc[0]['portfolio_max_dd'] == pytest.approx(expected)
