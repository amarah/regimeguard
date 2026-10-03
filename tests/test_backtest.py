"""Offline tests for backtest reporting."""

import math
import pandas as pd
import pytest


def test_metrics_include_the_first_simulated_day():
    from regimeguard.backtest import _performance_metrics

    curve = pd.Series([90.0, 99.0], index=pd.bdate_range("2026-01-02", periods=2))
    metrics = _performance_metrics(curve, 100.0)
    assert metrics["total_return"] == -0.01
    assert metrics["max_drawdown"] == -0.10
    assert all(math.isfinite(value) for value in metrics.values())


def test_metrics_validate_inputs():
    from regimeguard.backtest import _performance_metrics

    with pytest.raises(ValueError, match="positive initial capital"):
        _performance_metrics(pd.Series(dtype=float), 100)
    with pytest.raises(ValueError, match="positive initial capital"):
        _performance_metrics(pd.Series([100.0]), 0)
