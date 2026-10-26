"""Missing option data must not produce an insurance-pricing verdict."""
from types import SimpleNamespace

import pandas as pd
import pytest

from regimeguard import skew


def fake_ticker(monkeypatch, spot=100.0, put_iv=0.30, call_iv=0.20,
                expiry_days=30):
    expiry = (pd.Timestamp.now().normalize()
              + pd.Timedelta(days=expiry_days)).strftime("%Y-%m-%d")
    chain = SimpleNamespace(
        puts=pd.DataFrame({"strike": [92.0], "impliedVolatility": [put_iv]}),
        calls=pd.DataFrame({"strike": [108.0], "impliedVolatility": [call_iv]}),
    )
    ticker = SimpleNamespace(fast_info={"last_price": spot}, options=[expiry],
                             option_chain=lambda _: chain)
    monkeypatch.setattr(skew.yf, "Ticker", lambda _: ticker)


@pytest.mark.parametrize("side", ["put_iv", "call_iv"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 0.0])
def test_invalid_iv_has_no_verdict(monkeypatch, side, value):
    fake_ticker(monkeypatch, **{side: value})
    assert skew.analyze_skew("TEST") == {}


@pytest.mark.parametrize("spot", [float("nan"), float("inf"), -100.0, 0.0])
def test_invalid_spot_has_no_verdict(monkeypatch, spot):
    fake_ticker(monkeypatch, spot=spot)
    assert skew.analyze_skew("TEST") == {}


def test_valid_skew_is_preserved(monkeypatch):
    fake_ticker(monkeypatch)
    result = skew.analyze_skew("TEST")
    assert result["skew"] == 0.10
    assert result["put_iv_25d_proxy"] == 0.30
    assert result["call_iv_25d_proxy"] == 0.20
    assert "EXPENSIVE" in result["verdict"]


def test_twenty_calendar_day_expiry_is_included(monkeypatch):
    fake_ticker(monkeypatch, expiry_days=20)

    result = skew.analyze_skew("TEST")

    assert result["expiry"] == (pd.Timestamp.now().normalize()
                                + pd.Timedelta(days=20)).strftime("%Y-%m-%d")


@pytest.mark.parametrize("failure", ["expiries", "chain"])
def test_provider_failures_have_no_verdict(monkeypatch, failure):
    expiry = (pd.Timestamp.now().normalize()
              + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    class Ticker:
        fast_info = {"last_price": 100.0}

        @property
        def options(self):
            if failure == "expiries":
                raise RuntimeError("provider unavailable")
            return [expiry]

        def option_chain(self, _):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(skew.yf, "Ticker", lambda _: Ticker())

    assert skew.analyze_skew("TEST") == {}


@pytest.mark.parametrize(
    "puts,calls",
    [
        (pd.DataFrame({"strike": [92.0]}),
         pd.DataFrame({"strike": [108.0], "impliedVolatility": [0.20]})),
        (pd.DataFrame({"strike": [92.0], "impliedVolatility": ["missing"]}),
         pd.DataFrame({"strike": [108.0], "impliedVolatility": [0.20]})),
        (None, pd.DataFrame({"strike": [108.0], "impliedVolatility": [0.20]})),
    ],
)
def test_malformed_chain_has_no_verdict(monkeypatch, puts, calls):
    expiry = (pd.Timestamp.now().normalize()
              + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    chain = SimpleNamespace(puts=puts, calls=calls)
    ticker = SimpleNamespace(fast_info={"last_price": 100.0}, options=[expiry],
                             option_chain=lambda _: chain)
    monkeypatch.setattr(skew.yf, "Ticker", lambda _: ticker)

    assert skew.analyze_skew("TEST") == {}


def test_chain_without_sides_has_no_verdict(monkeypatch):
    expiry = (pd.Timestamp.now().normalize()
              + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    ticker = SimpleNamespace(fast_info={"last_price": 100.0}, options=[expiry],
                             option_chain=lambda _: SimpleNamespace())
    monkeypatch.setattr(skew.yf, "Ticker", lambda _: ticker)

    assert skew.analyze_skew("TEST") == {}
