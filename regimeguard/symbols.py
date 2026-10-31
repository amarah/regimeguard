"""Ticker-symbol normalization shared by user-facing entry points."""


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Canonicalize symbols and keep the first occurrence of each."""
    if not tickers:
        raise ValueError("At least one ticker symbol is required.")
    normalized = []
    seen = set()
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("Ticker symbols cannot be blank.")
        if symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    return normalized
