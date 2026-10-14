"""Data fetching with local parquet caching to avoid rate limits."""
import os
import time

import pandas as pd
import yfinance as yf

from .config import CACHE_DIR


def get_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Download adjusted closes, cached for 12 hours on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = "_".join(sorted(tickers)) + f"_{period}"
    path = os.path.join(CACHE_DIR, f"{key}.parquet")

    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if 0 <= age < 12 * 3600:
            return pd.read_parquet(path).dropna(how="all")

    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df = df.dropna(how="all")
    df.to_parquet(path)
    return df

