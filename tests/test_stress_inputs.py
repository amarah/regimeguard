"""Stress tests should fail clearly when portfolio weights are unusable."""
import math
from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard import stress


def test_empty_portfolio_is_rejected_before_downloading(monkeypatch):
    monkeypatch.setattr(
        stress, "get_prices",
        lambda *_args, **_kwargs: pytest.fail("download should not run"),
    )

    with pytest.raises(ValueError, match="At least one portfolio weight"):
        stress.run_stress_tests({})


@pytest.mark.parametrize("weight", [math.nan, math.inf, "missing", None])
def test_invalid_weight_is_rejected_before_downloading(monkeypatch, weight):
    monkeypatch.setattr(
        stress, "get_prices",
        lambda *_args, **_kwargs: pytest.fail("download should not run"),
    )

    with pytest.raises(ValueError, match="finite numbers"):
        stress.run_stress_tests({"SPY": weight})



def test_missing_requested_asset_is_reported_before_event_loop(monkeypatch):
    monkeypatch.setattr(
        stress,
        "get_prices",
        lambda *_args, **_kwargs: pd.DataFrame({"SPY": [100.0, 101.0]}),
    )

    with pytest.raises(ValueError, match="Missing price data for: AAPL"):
        stress.run_stress_tests({"AAPL": 1.0})


def test_stress_normalizes_and_combines_weight_symbols(monkeypatch):
    index = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({
        "AAPL": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "SPY": [200.0] * 6,
    }, index=index)
    download = Mock(return_value=prices)
    monkeypatch.setattr(stress, "get_prices", download)
    monkeypatch.setattr(
        stress, "STRESS_EVENTS", {"sample": (index[0], index[-1])}
    )

    result = stress.run_stress_tests({" aapl ": 0.25, "AAPL": 0.75})

    download.assert_called_once_with(["AAPL", "SPY"], period="max")
    assert result.loc[0, "portfolio_return"] == 0.05
