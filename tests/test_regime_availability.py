from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from regimeguard import regimes


@pytest.mark.parametrize('days', [0, 63, 65])
def test_short_history_fails_before_model_fit(monkeypatch, days):
    prices = pd.DataFrame({'SPY': np.arange(days) + 100.0})
    monkeypatch.setattr(regimes, 'get_prices', lambda *args: prices)
    model = Mock(side_effect=AssertionError('model must not be constructed'))
    monkeypatch.setattr(regimes, 'HAS_HMM', False)
    monkeypatch.setattr(regimes, 'GaussianMixture', model)
    with pytest.raises(ValueError, match='at least 3 complete feature rows'):
        regimes.detect_regime()


def test_unoccupied_states_report_unavailable_regime(monkeypatch):
    prices = pd.DataFrame({'SPY': np.arange(100) + 100.0})
    monkeypatch.setattr(regimes, 'get_prices', lambda *args: prices)
    model = Mock()
    model.predict.side_effect = lambda x: np.zeros(len(x), dtype=int)
    model.predict_proba.side_effect = lambda x: np.tile([1, 0, 0], (len(x), 1))
    monkeypatch.setattr(regimes, 'HAS_HMM', False)
    monkeypatch.setattr(regimes, 'GaussianMixture', lambda **kwargs: model)
    with pytest.raises(ValueError, match='did not populate all 3 states'):
        regimes.detect_regime()
