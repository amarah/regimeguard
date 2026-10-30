"""Data fetching with local parquet caching to avoid rate limits."""
import hashlib
import json
import os
import time
import tempfile

import pandas as pd
import yfinance as yf

from .config import CACHE_DIR


def _cache_path(tickers: list[str], period: str) -> str:
    """Build a stable filename without ambiguous separators or path input."""
    identity = json.dumps(
        {"period": period, "tickers": sorted(tickers)},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return os.path.join(CACHE_DIR, f"prices_{digest}.parquet")


def get_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Download adjusted closes, cached for 12 hours on disk."""
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("Provide at least one ticker.")
    if any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
        raise ValueError("Tickers must be nonempty strings.")
    if not isinstance(period, str) or not period.strip():
        raise ValueError("Period must be a nonempty string.")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(tickers, period)

    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if 0 <= age < 12 * 3600:
            try:
                cached = pd.read_parquet(path).dropna(how="all")
            except (OSError, ValueError):
                cached = pd.DataFrame()
            if not cached.empty:
                return cached

    downloaded = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    try:
        df = downloaded["Close"]
    except (KeyError, TypeError):
        raise ValueError(
            "No price data returned; the download was not cached. Try again later."
        ) from None
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    elif not isinstance(df, pd.DataFrame):
        raise ValueError(
            "No price data returned; the download was not cached. Try again later."
        )
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No price data returned; the download was not cached. Try again later.")
    # Publish only a complete Parquet file. A same-directory replacement is
    # atomic, so readers keep seeing the old cache until serialization succeeds.
    fd, temporary = tempfile.mkstemp(prefix=".prices-", suffix=".parquet", dir=CACHE_DIR)
    os.close(fd)
    try:
        df.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return df
