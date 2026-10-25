"""Ticker normalization prevents duplicate positions and collapsed weights."""
import pytest

from cli import normalize_tickers


def test_tickers_are_canonicalized_and_deduplicated_in_order():
    assert normalize_tickers(["aapl", " MSFT ", "AAPL", "msft", "BRK.B"]) == [
        "AAPL", "MSFT", "BRK.B"]


def test_blank_ticker_is_rejected():
    with pytest.raises(ValueError, match="cannot be blank"):
        normalize_tickers(["AAPL", " "])
