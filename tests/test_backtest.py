"""Offline tests for backtest reporting."""

import pandas as pd
import pytest
import math

from regimeguard.backtest import _performance_metrics


def test_metrics_include_the_first_simulated_day():
    curve = pd.Series([90.0, 99.0], index=pd.bdate_range("2026-01-02", periods=2))
    metrics = _performance_metrics(curve, 100.0)
    assert metrics["total_return"] == -0.01
    assert metrics["max_drawdown"] == -0.10
    assert all(math.isfinite(value) for value in metrics.values())


def test_metrics_validate_inputs():
    with pytest.raises(ValueError, match="positive initial capital"):
        _performance_metrics(pd.Series(dtype=float), 100)
    with pytest.raises(ValueError, match="positive initial capital"):
        _performance_metrics(pd.Series([100.0]), 0)
