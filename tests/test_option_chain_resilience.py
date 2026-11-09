from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from regimeguard import options


def test_expiry_boundary_uses_calendar_days(monkeypatch):
    today = pd.Timestamp("2026-11-09 23:45")
    assert options._days_to_expiry("2026-11-29", today) == 20


def test_failed_expiry_does_not_hide_other_chains(monkeypatch):
    today = pd.Timestamp.now().normalize()
    first = (today + pd.Timedelta(days=25)).strftime("%Y-%m-%d")
    second = (today + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    valid = pd.DataFrame({"strike": [90, 101], "lastPrice": [2.0, 1.0]})

    def chain(expiry):
        if expiry == first:
            raise RuntimeError("provider failed for one expiry")
        return SimpleNamespace(puts=valid)

    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": 100}, options=[first, second], option_chain=chain))

    result = options.fetch_put_chain("TEST")

    assert result["expiry"].tolist() == [second]
    assert result["dte"].tolist() == [30]
    assert result["strike"].tolist() == [90]


def test_invalid_spot_and_expiry_list_return_no_candidates(monkeypatch):
    chain = Mock(side_effect=AssertionError("option chain should not be requested"))
    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": float("nan")}, options=["2026-12-01"], option_chain=chain))
    assert options.fetch_put_chain("TEST").empty
    chain.assert_not_called()

    class BrokenTicker:
        fast_info = {"last_price": 100}

        @property
        def options(self):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(options.yf, "Ticker", lambda _: BrokenTicker())
    assert options.fetch_put_chain("TEST").empty
