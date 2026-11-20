"""Ticker-symbol normalization shared by user-facing entry points."""


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Canonicalize symbols and keep the first occurrence of each."""
    if not tickers:
        raise ValueError("At least one ticker symbol is required.")
    normalized = []
    seen = set()
    for ticker in tickers:
        if not isinstance(ticker, str):
            raise ValueError("Ticker symbols must be strings.")
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("Ticker symbols cannot be blank.")
        if symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    return normalized


def normalize_weight_map(weights: dict[str, float]) -> dict[str, float]:
    """Canonicalize portfolio symbols and combine duplicate entries."""
    normalized = {}
    for ticker, weight in weights.items():
        symbol = normalize_tickers([ticker])[0]
        normalized[symbol] = normalized.get(symbol, 0.0) + weight
    return normalized
