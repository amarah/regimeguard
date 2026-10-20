import pandas as pd

from regimeguard import clusters


def test_missing_prices_do_not_create_artificial_returns(monkeypatch):
    prices = pd.DataFrame({'A': [100, 110, None, 80, 100, 120],
                           'B': [100, 110, 80, 80, 100, 120]})
    monkeypatch.setattr(clusters, 'get_prices', lambda *args, **kwargs: prices)
    # The three observed one-day return pairs agree exactly. Filling the gap
    # invents a zero and a multi-day return, obscuring that relationship.
    assert clusters.correlation_clusters(['A', 'B']) == [['A', 'B']]
