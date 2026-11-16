"""Backtests need complete portfolio data and enough aligned returns."""
from unittest.mock import Mock

import pandas as pd
import pytest


def test_missing_requested_asset_has_clear_error(monkeypatch, tmp_path):
    import regimeguard.backtest as backtest

    prices = pd.DataFrame({"AAPL": [100, 101, 102]})
    monkeypatch.setattr(backtest, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(backtest, "get_prices", Mock(return_value=prices))

    with pytest.raises(ValueError, match="Missing price data for: MSFT, SPY"):
        backtest.run_backtest(["AAPL", "MSFT"], use_cached=False)


@pytest.mark.parametrize("prices", [
    pd.DataFrame({"AAPL": [100.0], "SPY": [200.0]}),
    pd.DataFrame({"AAPL": [100.0, 101.0], "SPY": [200.0, 201.0]}),
    pd.DataFrame({"AAPL": [100.0, None, 102.0], "SPY": [200.0, 201.0, 202.0]}),
])
def test_insufficient_aligned_history_is_rejected(monkeypatch, tmp_path, prices):
    import regimeguard.backtest as backtest

    monkeypatch.setattr(backtest, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(backtest, "get_prices", Mock(return_value=prices))

    with pytest.raises(ValueError, match="at least two aligned return observations"):
        backtest.run_backtest(["AAPL"], use_cached=False)
