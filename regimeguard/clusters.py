"""Redundant exposure detection via correlation clustering."""
import itertools

from .data import get_prices


def correlation_clusters(tickers: list[str], threshold: float = 0.75):
    corr = get_prices(tickers, period="2y").pct_change().dropna().corr()
    parent = {t: t for t in corr.columns}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(corr.columns, 2):
        if abs(corr.loc[a, b]) >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups = {}
    for t in corr.columns:
        groups.setdefault(find(t), []).append(t)
    return list(groups.values())
