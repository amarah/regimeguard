"""Malformed provider responses should fail cleanly and never be cached."""
from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard import data


@pytest.mark.parametrize(
    "payload",
    [None, pd.DataFrame(), pd.DataFrame({"Adj Close": [100.0]}), {"Close": 100.0}],
)
def test_missing_close_frame_is_rejected(monkeypatch, tmp_path, payload):
    monkeypatch.setattr(data, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(data.yf, "download", Mock(return_value=payload))

    with pytest.raises(ValueError, match="No price data"):
        data.get_prices(["AAPL"])

    assert not list(tmp_path.glob("*.parquet"))
