from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    "max_cost_pct", [float("nan"), float("inf"), None, "0.01", True]
)
def test_collar_rejects_invalid_cost_limit(max_cost_pct, monkeypatch):
    ticker = Mock(side_effect=AssertionError("provider should not be called"))
    monkeypatch.setattr(options.yf, "Ticker", ticker)

    with pytest.raises(ValueError, match="max_cost_pct must be a finite number"):
        options.optimize_collar(
            ["TEST"], {"TEST": 10_000}, max_cost_pct=max_cost_pct
        )

    ticker.assert_not_called()


@pytest.mark.parametrize(
    "min_floor", [-1, 0, 0.2, float("nan"), None, "-0.2", True]
)
def test_collar_rejects_invalid_protection_floor(min_floor, monkeypatch):
    ticker = Mock(side_effect=AssertionError("provider should not be called"))
    monkeypatch.setattr(options.yf, "Ticker", ticker)

    with pytest.raises(ValueError, match="min_floor must be between -1 and 0"):
        options.optimize_collar(
            ["TEST"], {"TEST": 10_000}, min_floor=min_floor
        )

    ticker.assert_not_called()
