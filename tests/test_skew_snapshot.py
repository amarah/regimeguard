from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from regimeguard import skew


def test_put_and_call_use_one_chain_snapshot(monkeypatch):
    expiry = (pd.Timestamp.now() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    chain = SimpleNamespace(
        puts=pd.DataFrame({'strike': [92], 'impliedVolatility': [0.3]}),
        calls=pd.DataFrame({'strike': [108], 'impliedVolatility': [0.2]}))
    fetch = Mock(return_value=chain)
    monkeypatch.setattr(skew.yf, 'Ticker', lambda _: SimpleNamespace(
        fast_info={'last_price': 100}, options=[expiry], option_chain=fetch))
    assert skew.analyze_skew('TEST')['skew'] == 0.1
    fetch.assert_called_once_with(expiry)
