"""Reject malformed price requests before touching cache or network state."""
from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard import data


@pytest.mark.parametrize("tickers", [None, (), [], ["AAPL", ""], ["AAPL", "   "], [123]])
def test_invalid_ticker_lists_are_rejected(monkeypatch, tmp_path, tickers):
    download = Mock()
    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(data.yf, "download", download)

    with pytest.raises(ValueError, match="(?i)ticker"):
        data.get_prices(tickers)

    download.assert_not_called()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("period", [None, "", "   ", 5])
def test_invalid_periods_are_rejected(monkeypatch, tmp_path, period):
    download = Mock()
    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(data.yf, "download", download)

    with pytest.raises(ValueError, match="Period"):
        data.get_prices(["AAPL"], period=period)

    download.assert_not_called()
    assert not (tmp_path / "cache").exists()


def test_duplicate_tickers_are_removed_before_download(monkeypatch, tmp_path):
    prices = pd.DataFrame({"AAPL": [100.0], "MSFT": [200.0]})
    download = Mock(return_value=pd.concat({"Close": prices}, axis=1))
    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data.yf, "download", download)

    result = data.get_prices(["AAPL", "MSFT", "AAPL"])

    pd.testing.assert_frame_equal(result, prices)
    download.assert_called_once_with(
        ["AAPL", "MSFT"], period="5y", auto_adjust=True, progress=False
    )
