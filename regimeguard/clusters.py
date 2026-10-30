"""Redundant exposure detection via correlation clustering."""
import itertools
from math import isfinite

from .data import get_prices


def correlation_clusters(tickers: list[str], threshold: float = 0.75):
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Correlation threshold must be a number from 0 to 1.")
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Correlation threshold must be between 0 and 1.")
    if not tickers:
        raise ValueError("At least one ticker is required for correlation clustering.")
    returns = get_prices(tickers, period="2y").pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError(
            "Correlation clustering needs at least two aligned return observations."
        )
    corr = returns.corr()
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
