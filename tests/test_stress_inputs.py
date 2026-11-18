"""Stress tests should fail clearly when portfolio weights are unusable."""
import math

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
