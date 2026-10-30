"""Correlation reports need enough aligned returns to estimate a relationship."""
import pandas as pd
import pytest

from regimeguard import clusters


@pytest.mark.parametrize(
    "prices",
    [
        pd.DataFrame({"A": [100.0], "B": [200.0]}),
        pd.DataFrame({"A": [100.0, 101.0], "B": [200.0, 202.0]}),
        pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [200.0, None, 202.0]}),
    ],
)
def test_insufficient_aligned_returns_are_rejected(monkeypatch, prices):
    monkeypatch.setattr(clusters, "get_prices", lambda *args, **kwargs: prices)

    with pytest.raises(ValueError, match="at least two aligned return observations"):
        clusters.correlation_clusters(["A", "B"])
