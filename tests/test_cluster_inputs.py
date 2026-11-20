"""Correlation clustering should reject settings that cannot be interpreted."""
import math
from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard import clusters


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan, math.inf])
def test_threshold_outside_correlation_range_is_rejected(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        clusters.correlation_clusters(["AAA", "BBB"], threshold)


@pytest.mark.parametrize("threshold", [None, "0.75", True])
def test_nonnumeric_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="number from 0 to 1"):
        clusters.correlation_clusters(["AAA", "BBB"], threshold)


def test_empty_ticker_list_is_rejected():
    with pytest.raises(ValueError, match="At least one ticker symbol"):
        clusters.correlation_clusters([])


def test_direct_api_normalizes_and_deduplicates_tickers(monkeypatch):
    prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]})
    download = Mock(return_value=prices)
    monkeypatch.setattr(clusters, "get_prices", download)

    result = clusters.correlation_clusters([" aapl ", "AAPL"])

    download.assert_called_once_with(["AAPL"], period="2y")
    assert result == [["AAPL"]]

