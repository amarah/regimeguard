from types import SimpleNamespace

import pandas as pd

from regimeguard import options


def _expiry(days=40):
    return (pd.Timestamp.now().normalize()
            + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def test_duplicate_strikes_use_best_available_quotes(monkeypatch):
    puts = pd.DataFrame({
        "strike": [95, 95, 90, 90],
        "ask": [4.0, 3.0, 1.5, 1.2],
        "bid": [3.5, 2.5, 1.0, 1.1],
    })
    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": "100"}, options=[_expiry()],
        option_chain=lambda _: SimpleNamespace(puts=puts)))

    result = options.build_put_spreads(
        ["TEST"], {"TEST": 10_000}, width_targets=(0.05,))

    assert result.loc[0, "spread"] == "95/90P"
    assert result.loc[0, "debit"] == 1.9
    assert result.loc[0, "max_gain_ps"] == 3.1
    assert result.loc[0, "protection_if_below_short_k"] == "3.1%"


def test_malformed_put_spread_quotes_are_ignored(monkeypatch):
    puts = pd.DataFrame({
        "strike": [95, 90],
        "ask": [float("inf"), 1.2],
        "bid": [3.5, "bad"],
    })
    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": 100}, options=[_expiry()],
        option_chain=lambda _: SimpleNamespace(puts=puts)))

    assert options.build_put_spreads(
        ["TEST"], {"TEST": 10_000}, width_targets=(0.05,)).empty


def test_put_spread_provider_failure_returns_no_candidates(monkeypatch):
    class BrokenTicker:
        fast_info = {"last_price": 100}
        options = [_expiry()]

        def option_chain(self, _):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(options.yf, "Ticker", lambda _: BrokenTicker())

    assert options.build_put_spreads(["TEST"], {"TEST": 10_000}).empty
