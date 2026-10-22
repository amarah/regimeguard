"""Regime features retain their names through model fitting and reporting."""
import numpy as np
import pandas as pd
import pytest

from regimeguard import regimes


@pytest.mark.parametrize("use_hmm", [False, True])
def test_named_features_and_state_statistics(monkeypatch, use_hmm):
    index = pd.bdate_range("2024-01-01", periods=150)
    returns = np.resize([0.01, -0.02, 0.003], len(index))
    prices = pd.DataFrame({"SPY": 100 * np.cumprod(1 + returns)}, index=index)
    prices.loc[index[80], "SPY"] = np.nan
    ret = prices.SPY.pct_change(fill_method=None).rename("ret")
    expected = pd.concat([
        ret,
        (ret.rolling(21).std() * np.sqrt(regimes.TRADING_DAYS)).rename("vol"),
        prices.SPY.pct_change(63, fill_method=None).rename("mom"),
    ], axis=1).dropna()
    states = np.resize([2, 0, 1], len(expected))
    probabilities = np.tile([0.15, 0.25, 0.60], (len(expected), 1))

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["n_components"] == 3

        def fit(self, values):
            np.testing.assert_allclose(values, expected.values)

        def predict(self, values):
            return states

        def predict_proba(self, values):
            return probabilities

    monkeypatch.setattr(regimes, "get_prices", lambda *args: prices)
    monkeypatch.setattr(regimes, "HAS_HMM", use_hmm)
    monkeypatch.setattr(regimes, "GaussianHMM", Model, raising=False)
    monkeypatch.setattr(regimes, "GaussianMixture", Model)
    frame, state = regimes.detect_regime()

    pd.testing.assert_frame_equal(frame[["ret", "vol", "mom"]], expected)
    assert set(state.history) == {"RISK-ON", "CHOPPY", "RISK-OFF"}
    current = states[-1]
    selected = expected.iloc[states == current]
    assert state.probability == probabilities[-1, current]
    assert state.ann_vol == round(selected.vol.mean(), 4)
    assert state.ann_trend == round(selected.ret.mean() * regimes.TRADING_DAYS, 4)
    means = expected.assign(state=states).groupby("state")["ret"].mean().sort_values(
        ascending=False
    )
    label_map = {
        means.index[0]: "RISK-ON",
        means.index[1]: "CHOPPY",
        means.index[2]: "RISK-OFF",
    }
    assert state.label == label_map[current]
    assert state.position_multiplier == regimes.MULTIPLIERS[label_map[current]]
    pd.testing.assert_index_equal(state.history.index, expected.index)
