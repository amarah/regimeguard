import numpy as np


def test_kelly_bounds():
    from regimeguard.sizing import kelly_fraction
    assert kelly_fraction(0.6, 2.0) > 0
    assert kelly_fraction(0.3, 1.0) == 0
    assert kelly_fraction(0.9, 5.0) <= 0.5


def test_mc_cvar_worse_than_var():
    from regimeguard.risk import monte_carlo_cvar
    res = monte_carlo_cvar({"AAPL": 1.0}, sims=5000)
    assert res["CVaR_95"] >= res["VaR_95"]
    assert res["prob_loss"] > 0.3


def test_black_scholes_put_bounds():
    from regimeguard.options import black_scholes_put
    p = black_scholes_put(100, 100, T=30/365, sigma=0.25)
    assert 0 < p < 10
    assert black_scholes_put(90, 100, T=30/365) > black_scholes_put(110, 100, T=30/365)


def test_correlation_clustering():
    from regimeguard.clusters import correlation_clusters
    groups = correlation_clusters(["AAPL", "MSFT", "GLD"])
    flat = [t for g in groups for t in g]
    assert sorted(flat) == ["AAPL", "GLD", "MSFT"]
