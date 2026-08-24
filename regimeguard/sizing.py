"""Fractional Kelly position sizing with volatility targeting."""
import numpy as np
import pandas as pd

from .config import MAX_POSITION_WEIGHT, PORTFOLIO_VOL_TARGET, TRADING_DAYS
from .data import get_prices


def kelly_fraction(win_rate: float, payoff: float, fraction: float = 0.5) -> float:
    b = max(payoff, 1e-9)
    f = win_rate - (1 - win_rate) / b
    return float(np.clip(fraction * f, 0.0, 0.5))


def estimate_trade_stats(rets: pd.Series, lookback: int = 126):
    r = rets.iloc[-lookback:].dropna()
    wins = r[r > 0]
    losses = r[r <= 0]
    wr = len(wins) / len(r) if len(r) else 0.5
    avg_win = wins.mean() if len(wins) else 0.01
    avg_loss = abs(losses.mean()) if len(losses) else 0.01
    return wr, avg_win / max(avg_loss, 1e-9)


def size_positions(tickers: list[str], capital: float,
                   regime_mult: float) -> pd.DataFrame:
    prices = get_prices(tickers, period="2y")
    rows = []
    for t in tickers:
        px = prices[t].dropna()
        if px.empty:
            continue
        rets = px.pct_change().dropna()
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
