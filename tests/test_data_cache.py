"""Exercise the real Parquet cache using synthetic prices."""
import os
import time
from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard import data


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'CACHE_DIR', str(tmp_path))
    return tmp_path


def prices():
    return pd.DataFrame({'AAPL': [100., None, 102.], 'MSFT': [200., None, None]},
                        index=pd.date_range('2026-01-05', periods=3))


def test_download_and_cache_hit_return_same_rows(monkeypatch, cache):
    raw = prices()
    download = Mock(return_value=pd.concat({'Close': raw}, axis=1))
    monkeypatch.setattr(data.yf, 'download', download)
    first = data.get_prices(['AAPL', 'MSFT'])
    second = data.get_prices(['MSFT', 'AAPL'])
    pd.testing.assert_frame_equal(first, raw.dropna(how='all'), check_freq=False)
    pd.testing.assert_frame_equal(second, first, check_freq=False)
    download.assert_called_once()
    assert len(second) == 2  # Keep partially populated rows, not all-null rows.


def test_existing_cache_with_empty_rows_is_cleaned(monkeypatch, cache):
    prices().to_parquet(cache / 'AAPL_MSFT_5y.parquet')
    download = Mock(side_effect=AssertionError('fresh cache should avoid downloads'))
    monkeypatch.setattr(data.yf, 'download', download)
    pd.testing.assert_frame_equal(data.get_prices(['AAPL', 'MSFT']), prices().dropna(how='all'), check_freq=False)


def test_corrupt_fresh_cache_is_replaced(monkeypatch, cache):
    path = cache / 'AAPL_MSFT_5y.parquet'
    path.write_bytes(b'not a parquet file')
    fresh = prices() * 2
    download = Mock(return_value=pd.concat({'Close': fresh}, axis=1))
    monkeypatch.setattr(data.yf, 'download', download)

    result = data.get_prices(['AAPL', 'MSFT'])

    pd.testing.assert_frame_equal(result, fresh.dropna(how='all'), check_freq=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result, check_freq=False)
    download.assert_called_once()


def test_stale_cache_downloads_replacement(monkeypatch, cache):
    path = cache / 'AAPL_MSFT_5y.parquet'
    prices().to_parquet(path)
    stale = time.time() - 13 * 3600
    os.utime(path, (stale, stale))
    fresh = prices() * 2
    download = Mock(return_value=pd.concat({'Close': fresh}, axis=1))
    monkeypatch.setattr(data.yf, 'download', download)
    pd.testing.assert_frame_equal(data.get_prices(['AAPL', 'MSFT']), fresh.dropna(how='all'), check_freq=False)
    download.assert_called_once()


def test_single_ticker_series_is_named_and_cached(monkeypatch, cache):
    raw = prices()['AAPL']
    monkeypatch.setattr(data.yf, 'download', Mock(return_value=pd.DataFrame({'Close': raw})))
    expected = raw.dropna().to_frame('AAPL')
    pd.testing.assert_frame_equal(data.get_prices(['AAPL']), expected, check_freq=False)
    pd.testing.assert_frame_equal(data.get_prices(['AAPL']), expected, check_freq=False)


@pytest.mark.parametrize('age', [12 * 3600, -60])
def test_expiry_boundary_and_future_timestamp_refresh(monkeypatch, cache, age):
    path = cache / 'AAPL_MSFT_5y.parquet'
    prices().to_parquet(path)
    modified = path.stat().st_mtime
    monkeypatch.setattr(data.time, 'time', lambda: modified + age)
    download = Mock(return_value=pd.concat({'Close': prices()}, axis=1))
    monkeypatch.setattr(data.yf, 'download', download)
    data.get_prices(['AAPL', 'MSFT'])
    download.assert_called_once()
