"""Correlation clustering should reject settings that cannot be interpreted."""
import math

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
    with pytest.raises(ValueError, match="At least one ticker"):
        clusters.correlation_clusters([])

