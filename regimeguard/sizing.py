"""Fractional Kelly position sizing with volatility targeting."""
from numbers import Real

import numpy as np
import pandas as pd

from .config import MAX_POSITION_WEIGHT, PORTFOLIO_VOL_TARGET, TRADING_DAYS
from .data import get_prices
from .symbols import normalize_tickers


def kelly_fraction(win_rate: float, payoff: float, fraction: float = 0.5) -> float:
    values = {
        "Win rate": win_rate,
        "Payoff": payoff,
        "Kelly fraction": fraction,
    }
    for label, value in values.items():
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value)):
            raise ValueError(f"{label} must be a finite number.")
    if not 0 <= win_rate <= 1:
        raise ValueError("Win rate must be between 0 and 1.")
    if payoff <= 0:
        raise ValueError("Payoff must be positive.")
    if not 0 <= fraction <= 1:
        raise ValueError("Kelly fraction must be between 0 and 1.")
    b = max(payoff, 1e-9)
    f = win_rate - (1 - win_rate) / b
    return float(np.clip(fraction * f, 0.0, 0.5))


def estimate_trade_stats(rets: pd.Series, lookback: int = 126):
    r = rets.iloc[-lookback:].dropna()
    wins = r[r > 0]
    losses = r[r < 0]
    decided = len(wins) + len(losses)
    wr = len(wins) / decided if decided else 0.5
    avg_win = wins.mean() if len(wins) else 0.01
    avg_loss = abs(losses.mean()) if len(losses) else 0.01
    return wr, avg_win / max(avg_loss, 1e-9)


def size_positions(tickers: list[str], capital: float,
                   regime_mult: float) -> pd.DataFrame:
    try:
        capital = float(capital)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Capital must be a positive finite number.") from None
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("Capital must be a positive finite number.")
    try:
        regime_mult = float(regime_mult)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Regime multiplier must be a nonnegative finite number.") from None
    if not np.isfinite(regime_mult) or regime_mult < 0:
        raise ValueError("Regime multiplier must be a nonnegative finite number.")
    tickers = normalize_tickers(tickers)
    prices = get_prices(tickers, period="2y")
    rows = []
    for t in tickers:
        series = prices[t]
        px = series.dropna()
        if px.empty:
            continue
        # A return is usable only when both adjacent trading-day quotes exist.
        # Dropping prices first would compress gaps and treat a multi-day move
        # as a one-day observation, overstating volatility and trade frequency.
        rets = series.pct_change(fill_method=None).dropna()
        if len(rets) < 2:
            continue
        wr, blr = estimate_trade_stats(rets)
        weight = kelly_fraction(wr, blr) * regime_mult
        vol = float(rets.std() * np.sqrt(TRADING_DAYS))
        weight *= min(PORTFOLIO_VOL_TARGET / max(vol, 1e-6), 1.0)
        weight = min(weight, MAX_POSITION_WEIGHT)
        price = float(px.iloc[-1])
        dollars = capital * weight
        rows.append({
            "ticker": t,
            "price": round(price, 2),
            "win_rate": round(wr, 3),
            "payoff": round(blr, 2),
            "ann_vol": round(vol, 4),
            "weight": round(weight, 4),
            "dollars": round(dollars, 2),
            "shares": int(dollars // price),
        })
    return pd.DataFrame(rows)
