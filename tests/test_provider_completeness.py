"""Portfolio reports must not silently discard assets missing from downloads."""
import pandas as pd
import pytest

from regimeguard import clusters, risk


def partial_prices():
    return pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]})


def test_risk_rejects_missing_portfolio_assets(monkeypatch):
    monkeypatch.setattr(risk, "get_prices", lambda *args, **kwargs: partial_prices())

    with pytest.raises(ValueError, match="Missing price data for: MSFT"):
        risk.monte_carlo_cvar({"AAPL": 0.5, "MSFT": 0.5}, sims=10)


def test_clusters_reject_missing_requested_assets(monkeypatch):
    monkeypatch.setattr(clusters, "get_prices", lambda *args, **kwargs: partial_prices())

    with pytest.raises(ValueError, match="Missing price data for: MSFT"):
        clusters.correlation_clusters(["AAPL", "MSFT"])
