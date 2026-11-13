from types import SimpleNamespace

import pandas as pd

from regimeguard import options


def _expiry(days=40):
    return (pd.Timestamp.now().normalize()
            + pd.Timedelta(days=days)).strftime("%Y-%m-%d")


def test_collar_fetches_each_chain_once(monkeypatch):
    calls = []
    chain = SimpleNamespace(
        puts=pd.DataFrame({"strike": [90], "ask": [2.0]}),
        calls=pd.DataFrame({"strike": [110], "bid": [2.0]}),
    )

    def option_chain(expiry):
        calls.append(expiry)
        return chain

    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": "100"}, options=[_expiry()],
        option_chain=option_chain))

    result = options.optimize_collar(["TEST"], {"TEST": 10_000})

    assert calls == [_expiry()]
    assert result.loc[0, "buy_put"] == 90
    assert result.loc[0, "sell_call"] == 110


def test_malformed_collar_quotes_are_ignored(monkeypatch):
    chain = SimpleNamespace(
        puts=pd.DataFrame({"strike": [90, 91], "ask": [float("inf"), "bad"]}),
        calls=pd.DataFrame({"strike": [110], "bid": [2.0]}),
    )
    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": 100}, options=[_expiry()],
        option_chain=lambda _: chain))

    assert options.optimize_collar(["TEST"], {"TEST": 10_000}).empty


def test_collar_provider_failures_return_no_candidates(monkeypatch):
    class BrokenTicker:
        fast_info = {"last_price": 100}
        options = [_expiry()]

        def option_chain(self, _):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(options.yf, "Ticker", lambda _: BrokenTicker())

    assert options.optimize_collar(["TEST"], {"TEST": 10_000}).empty


def test_collar_uses_expiry_nearest_thirty_days(monkeypatch):
    selected = []
    expiries = [_expiry(55), _expiry(40), _expiry(31)]
    chain = SimpleNamespace(
        puts=pd.DataFrame({"strike": [90], "ask": [2.0]}),
        calls=pd.DataFrame({"strike": [110], "bid": [2.0]}),
    )
    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": 100}, options=expiries,
        option_chain=lambda expiry: selected.append(expiry) or chain))

    result = options.optimize_collar(["TEST"], {"TEST": 10_000})

    assert selected == [_expiry(31)]
    assert result.loc[0, "expiry"] == _expiry(31)
