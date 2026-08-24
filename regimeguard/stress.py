"""Historical crisis replay against user's actual weights."""
import pandas as pd

from .data import get_prices
from .config import STRESS_EVENTS


def run_stress_tests(weights: dict[str, float]) -> pd.DataFrame:
    prices = get_prices(list(weights.keys()) + ["SPY"], period="max")
    rows = []
    for name, (start, end) in STRESS_EVENTS.items():
        try:
            window = prices.loc[start:end]
            if len(window) < 5:
                continue
            rets = window.pct_change().dropna()
            port_ret = (rets[list(weights.keys())] * pd.Series(weights)).sum(axis=1)
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
