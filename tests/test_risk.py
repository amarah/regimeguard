"""Offline unit tests — no network required."""
import numpy as np
import pandas as pd
import pytest


def _fake_prices(tickers=("AAPL",), days=400, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-01", periods=days)
    df = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, days))
         for t in tickers}, index=idx)
    return df


def test_kelly_bounds():
    from regimeguard.sizing import kelly_fraction
    assert kelly_fraction(0.6, 2.0) > 0
    assert kelly_fraction(0.3, 1.0) == 0
    assert kelly_fraction(0.9, 5.0) <= 0.5


def test_kelly_monotonic_in_edge():
    from regimeguard.sizing import kelly_fraction
    assert kelly_fraction(0.7, 2.0) > kelly_fraction(0.55, 2.0)


def test_black_scholes_put_bounds():
    from regimeguard.options import black_scholes_put
    p = black_scholes_put(100, 100, T=30/365, sigma=0.25)
    assert 0 < p < 10
    assert black_scholes_put(90, 100, T=30/365) > black_scholes_put(110, 100, T=30/365)


def test_mc_cvar_worse_than_var(monkeypatch):
    """Patch get_prices so no network call happens."""
    import regimeguard.risk as risk_mod

    fake = _fake_prices(("AAPL",))
    monkeypatch.setattr(risk_mod, "get_prices",
                        lambda tickers, period="3y": fake)

    res = risk_mod.monte_carlo_cvar({"AAPL": 1.0}, sims=5000)
    assert res["CVaR_95"] >= res["VaR_95"]
    assert res["prob_loss"] > 0


def test_correlation_clustering(monkeypatch):
    import regimeguard.clusters as clusters_mod

    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.01, 500)
    df = pd.DataFrame({
        "AAPL": 100 * np.cumprod(1 + common),
        "MSFT": 100 * np.cumprod(1 + common * 0.98 + rng.normal(0, 0.001, 500)),
        "GLD": 100 * np.cumprod(1 + rng.normal(0, 0.01, 500)),
    }, index=pd.bdate_range("2023-01-01", periods=500))

    monkeypatch.setattr(clusters_mod, "get_prices",
                        lambda tickers, period="2y": df)

    groups = clusters_mod.correlation_clusters(["AAPL", "MSFT", "GLD"])
    flat = sorted(t for g in groups for t in g)
    assert flat == ["AAPL", "GLD", "MSFT"]
    aapl_group = next(g for g in groups if "AAPL" in g)
    assert "MSFT" in aapl_group and "GLD" not in aapl_group
