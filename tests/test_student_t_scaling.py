"""Fat-tailed shocks should preserve the covariance estimated from returns."""
import numpy as np
import pandas as pd

from regimeguard import risk


class FixedRandom:
    def standard_normal(self, shape):
        return -np.ones(shape)

    def chisquare(self, degrees, size):
        return np.full(size, degrees, dtype=float)


def test_student_t_shocks_are_standardized_to_unit_variance(monkeypatch):
    prices = pd.DataFrame({"A": [100.0, 90.0, 99.0]})
    monkeypatch.setattr(risk, "get_prices", lambda *args, **kwargs: prices)
    monkeypatch.setattr(risk.np.random, "default_rng", lambda seed: FixedRandom())

    result = risk.monte_carlo_cvar({"A": 1.0}, horizon_days=1, sims=2,
                                   student_t=True)

    sample_vol = prices.pct_change().dropna()["A"].std()
    expected_loss = sample_vol * np.sqrt((5 - 2) / 5)
    assert result["VaR_95"] == round(expected_loss, 4)
    assert result["CVaR_95"] == round(expected_loss, 4)
