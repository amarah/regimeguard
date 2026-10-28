"""A failed refresh must never publish a partial cache file."""
import os
import time
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from regimeguard import data


@pytest.mark.parametrize('existing', [False, True])
def test_failed_write_preserves_previous_cache(monkeypatch, tmp_path, existing):
    monkeypatch.setattr(data, 'CACHE_DIR', str(tmp_path))
    path = Path(data._cache_path(['AAPL'], '5y'))
    old = pd.DataFrame({'AAPL': [90.]})
    if existing:
        old.to_parquet(path)
        stale = time.time() - 13 * 3600
        os.utime(path, (stale, stale))
    original = path.read_bytes() if existing else None
    fresh = pd.DataFrame({'AAPL': [100.]})
    monkeypatch.setattr(data.yf, 'download', Mock(return_value=pd.concat({'Close': fresh}, axis=1)))

    def interrupted_write(frame, destination, **kwargs):
        Path(destination).write_bytes(b'partial parquet')
        raise OSError('simulated disk failure')

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', interrupted_write)
    with pytest.raises(OSError, match='simulated disk failure'):
        data.get_prices(['AAPL'])
    if existing:
        assert path.read_bytes() == original
    else:
        assert not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ([path.name] if existing else [])


def test_complete_cache_is_published_only_after_serialization(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'CACHE_DIR', str(tmp_path))
    path = Path(data._cache_path(['AAPL'], '5y'))
    fresh = pd.DataFrame({'AAPL': [100., 101.]})
    monkeypatch.setattr(data.yf, 'download', Mock(return_value=pd.concat({'Close': fresh}, axis=1)))
    original_write = pd.DataFrame.to_parquet

    def inspect_write(frame, destination, **kwargs):
        assert Path(destination).parent == tmp_path
        assert Path(destination) != path
        assert not path.exists()
        return original_write(frame, destination, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'to_parquet', inspect_write)
    data.get_prices(['AAPL'])
    pd.testing.assert_frame_equal(pd.read_parquet(path), fresh)
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_cleans_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(data, 'CACHE_DIR', str(tmp_path))
    fresh = pd.DataFrame({'AAPL': [100.]})
    monkeypatch.setattr(data.yf, 'download', Mock(return_value=pd.concat({'Close': fresh}, axis=1)))
    monkeypatch.setattr(data.os, 'replace', Mock(side_effect=OSError('replacement failed')))
    with pytest.raises(OSError, match='replacement failed'):
        data.get_prices(['AAPL'])
    assert list(tmp_path.iterdir()) == []
