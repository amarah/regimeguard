"""Ticker normalization prevents duplicate positions and collapsed weights."""
import pytest

from regimeguard.symbols import normalize_tickers


def test_tickers_are_canonicalized_and_deduplicated_in_order():
    assert normalize_tickers(["aapl", " MSFT ", "AAPL", "msft", "BRK.B"]) == [
        "AAPL", "MSFT", "BRK.B"]


def test_blank_ticker_is_rejected():
    with pytest.raises(ValueError, match="cannot be blank"):
        normalize_tickers(["AAPL", " "])


def test_empty_ticker_list_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        normalize_tickers([])


@pytest.mark.parametrize("ticker", [None, 123, True, object()])
def test_non_string_ticker_is_rejected_cleanly(ticker):
    with pytest.raises(ValueError, match="must be strings"):
        normalize_tickers(["AAPL", ticker])
