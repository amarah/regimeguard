import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import cli


@pytest.mark.parametrize('json_output', [True, False])
def test_json_skips_unused_analyses(monkeypatch, capsys, json_output):
    monkeypatch.setattr('sys.argv', ['regimeguard', '--tickers', 'A'] + (['--json'] if json_output else []))
    monkeypatch.setattr(cli, 'detect_regime', lambda: (None, SimpleNamespace(
        label='CHOPPY', probability=0.8, position_multiplier=0.6, ann_trend=0.05, ann_vol=0.2)))
    monkeypatch.setattr(cli, 'size_positions', lambda *args: pd.DataFrame(
        {'ticker': ['A'], 'dollars': [100.0]}))
    risk = {'VaR_95': 0.1, 'CVaR_95': 0.2, 'DaR_95': 0.3, 'prob_loss': 0.4}
    monkeypatch.setattr(cli, 'monte_carlo_cvar', lambda weights: risk)
    stress = Mock(return_value=pd.DataFrame())
    clusters = Mock(return_value=[['A']])
    monkeypatch.setattr(cli, 'run_stress_tests', stress)
    monkeypatch.setattr(cli, 'correlation_clusters', clusters)
    cli.main()
    output = capsys.readouterr().out
    if json_output:
        result = json.loads(output)
        assert result['regime'] == 'CHOPPY'
        assert result['positions'] == [{'ticker': 'A', 'dollars': 100.0}]
        assert result['risk'] == risk
        stress.assert_not_called()
        clusters.assert_not_called()
    else:
        assert 'EXPOSURE CLUSTERS' in output
        stress.assert_called_once_with({'A': 1.0})
        clusters.assert_called_once_with(['A'])
