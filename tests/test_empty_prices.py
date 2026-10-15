from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard import data


@pytest.mark.parametrize('empty', [pd.DataFrame({'AAPL': []}),
                                  pd.DataFrame({'AAPL': [float('nan')]})])
def test_empty_download_is_not_cached(monkeypatch, tmp_path, empty):
    monkeypatch.setattr(data, 'CACHE_DIR', str(tmp_path))
    good = pd.DataFrame({'AAPL': [100.]})
    download = Mock(side_effect=[pd.concat({'Close': empty}, axis=1),
                                 pd.concat({'Close': good}, axis=1)])
    monkeypatch.setattr(data.yf, 'download', download)
    with pytest.raises(ValueError, match='No price data'):
        data.get_prices(['AAPL'])
    assert not list(tmp_path.glob('*.parquet'))
    pd.testing.assert_frame_equal(data.get_prices(['AAPL']), good)
    assert download.call_count == 2


def test_legacy_empty_cache_is_refetched(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'CACHE_DIR', str(tmp_path))
    pd.DataFrame({'AAPL': [float('nan')]}).to_parquet(tmp_path / 'AAPL_5y.parquet')
    good = pd.DataFrame({'AAPL': [100.]})
    download = Mock(return_value=pd.concat({'Close': good}, axis=1))
    monkeypatch.setattr(data.yf, 'download', download)
    pd.testing.assert_frame_equal(data.get_prices(['AAPL']), good)
    download.assert_called_once()
