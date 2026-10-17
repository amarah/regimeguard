"""Synthetic return series isolate the clustering rules without downloads."""
import numpy as np
import pandas as pd

from regimeguard import clusters


def run(monkeypatch, returns, threshold=0.75):
    # Include starting equity so every synthetic return enters pct_change.
    prices = pd.concat([pd.DataFrame({name: [100.] for name in returns}),
                        100 * (1 + pd.DataFrame(returns)).cumprod()], ignore_index=True)
    monkeypatch.setattr(clusters, 'get_prices', lambda *args, **kwargs: prices)
    return {frozenset(group) for group in clusters.correlation_clusters(list(returns), threshold)}


def test_inverse_returns_share_a_cluster(monkeypatch):
    values = np.array([-.02, -.01, .01, .02])
    assert run(monkeypatch, {'LONG': values, 'INVERSE': -values}) == {
        frozenset({'LONG', 'INVERSE'})}


def test_transitive_links_form_one_group(monkeypatch):
    # A and C are uncorrelated; B correlates with each at 1/sqrt(2).
    a = np.array([-1., -1., 1., 1.]) * .01
    c = np.array([-1., 1., -1., 1.]) * .01
    assert run(monkeypatch, {'A': a, 'B': a + c, 'C': c}, .7) == {
        frozenset({'A', 'B', 'C'})}
    assert run(monkeypatch, {'A': a, 'B': a + c, 'C': c}, .8) == {
        frozenset({'A'}), frozenset({'B'}), frozenset({'C'})}


def test_constant_series_does_not_create_spurious_links(monkeypatch):
    assert run(monkeypatch, {'FLAT': [0., 0., 0., 0.],
                            'MOVING': [-.02, .01, -.01, .02]}) == {
        frozenset({'FLAT'}), frozenset({'MOVING'})}


def test_single_symbol_is_preserved(monkeypatch):
    assert run(monkeypatch, {'ONLY': [.01, -.01, .02]}) == {frozenset({'ONLY'})}
