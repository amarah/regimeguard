"""Position sizing should reject inputs that cannot produce valid allocations."""
from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.mark.parametrize("capital", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_capital_is_rejected_before_download(monkeypatch, capital):
    import regimeguard.sizing as sizing

    download = Mock()
    monkeypatch.setattr(sizing, "get_prices", download)

    with pytest.raises(ValueError, match="Capital must be a positive finite number"):
        sizing.size_positions(["AAPL"], capital, 1.0)

    download.assert_not_called()


@pytest.mark.parametrize("multiplier", [-0.1, float("nan"), float("inf"), "bad"])
def test_invalid_regime_multiplier_is_rejected_before_download(monkeypatch, multiplier):
    import regimeguard.sizing as sizing

    download = Mock()
    monkeypatch.setattr(sizing, "get_prices", download)

    with pytest.raises(ValueError, match="Regime multiplier must be a nonnegative finite number"):
        sizing.size_positions(["AAPL"], 10_000, multiplier)

    download.assert_not_called()


def test_direct_api_normalizes_and_deduplicates_tickers(monkeypatch):
    import regimeguard.sizing as sizing

    prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]})
    download = Mock(return_value=prices)
    monkeypatch.setattr(sizing, "get_prices", download)

    result = sizing.size_positions([" aapl ", "AAPL"], 10_000, 1.0)

    download.assert_called_once_with(["AAPL"], period="2y")
    assert result["ticker"].tolist() == ["AAPL"]
