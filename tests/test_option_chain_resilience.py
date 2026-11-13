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
    valid = pd.DataFrame({"strike": [90, 101], "ask": [2.0, 1.0]})

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


def test_malformed_quotes_are_excluded_from_put_candidates(monkeypatch):
    expiry = (pd.Timestamp.now().normalize()
              + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    quotes = pd.DataFrame({
        "strike": [90, 91, "missing", 93, 94],
        "ask": [2.0, "missing", 1.0, float("inf"), 0.0],
    })
    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": 100}, options=[expiry],
        option_chain=lambda _: SimpleNamespace(puts=quotes)))

    result = options.fetch_put_chain("TEST")

    assert result["strike"].tolist() == [90.0]
    assert result["ask"].tolist() == [2.0]


def test_chain_without_ask_is_skipped(monkeypatch):
    expiry = (pd.Timestamp.now().normalize()
              + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    monkeypatch.setattr(options.yf, "Ticker", lambda _: SimpleNamespace(
        fast_info={"last_price": 100}, options=[expiry],
        option_chain=lambda _: SimpleNamespace(
            puts=pd.DataFrame({"strike": [90]}))))

    assert options.fetch_put_chain("TEST").empty


def test_hedge_cost_uses_executable_ask_quote(monkeypatch):
    monkeypatch.setattr(options, "fetch_put_chain", lambda _: pd.DataFrame({
        "strike": [90.0], "lastPrice": [1.0], "ask": [2.5],
        "spot": [100.0], "expiry": ["2026-12-18"], "dte": [35],
    }))

    result = options.optimize_hedge(
        ["TEST"], {"TEST": 10_000}, cvar_pct=0.2, top_n=1)

    assert result.loc[0, "premium"] == 2.5
    assert result.loc[0, "cost_pct"] == 0.025
