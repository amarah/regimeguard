"""Historical crisis replay against user's actual weights."""
import numpy as np
import pandas as pd

from .data import get_prices
from .config import STRESS_EVENTS


def run_stress_tests(weights: dict[str, float]) -> pd.DataFrame:
    if not weights:
        raise ValueError("At least one portfolio weight is required for stress testing.")
    try:
        numeric_weights = pd.Series(weights, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Portfolio weights must be finite numbers.") from None
    if not np.isfinite(numeric_weights).all():
        raise ValueError("Portfolio weights must be finite numbers.")
    tickers = list(weights.keys())
    requested = tickers if "SPY" in weights else tickers + ["SPY"]
    prices = get_prices(requested, period="max")
    rows = []
    for name, (start, end) in STRESS_EVENTS.items():
        try:
            window = prices.loc[start:end]
            if len(window) < 5:
                continue
            # Filling a missing quote creates a zero return followed by a
            # multi-day return. Use only genuinely adjacent, aligned prices.
            rets = window.pct_change(fill_method=None).dropna(
                subset=list(dict.fromkeys(tickers + ["SPY"])))
            if rets.empty:
                continue
            port_ret = (rets[tickers] * numeric_weights).sum(axis=1)
            total = (1 + port_ret).prod() - 1
            equity = (1 + port_ret).cumprod()
            max_dd = ((equity - equity.cummax()) / equity.cummax()).min()
            spy_total = (1 + rets["SPY"].fillna(0)).prod() - 1
            rows.append({
                "event": name,
                "portfolio_return": round(total, 4),
                "portfolio_max_dd": round(max_dd, 4),
                "spy_return": round(spy_total, 4),
                "vs_spy": round(total - spy_total, 4),
                "days_to_recover_estimate": int(abs(max_dd) / 0.002),
            })
        except KeyError:
            continue
    return pd.DataFrame(rows)
