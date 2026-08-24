"""
Full Strategy Backtester — simulates the complete RegimeGuard loop:
regime multiplier × Kelly × vol-target weights, monthly rebalance,
optional hedge drag. Benchmarks: equal-weight B&H and SPY.
Note: uses full-history regime fit (fast); strict walk-forward refitting
is a roadmap item.
"""
import os
import pickle

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, CACHE_DIR
from .data import get_prices
from .regimes import detect_regime


def _fit_regime_series(spy_rets: pd.Series) -> pd.Series:
    _, state = detect_regime()
    mult_map = {"RISK-ON": 1.0, "CHOPPY": 0.6, "RISK-OFF": 0.25}
    return state.history.map(mult_map)


def run_backtest(tickers: list[str], capital: float = 100_000,
                 period: str = "10y", hedge_drag_annual: float = 0.015,
                 use_cached: bool = True) -> dict:
    cache_path = os.path.join(
        CACHE_DIR, f"bt_{'_'.join(sorted(tickers))}_{period}.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)

    prices = get_prices(tickers + ["SPY"], period=period)
    rets = prices[tickers].pct_change().dropna()

    if use_cached and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            mult = pickle.load(f)
    else:
        mult = _fit_regime_series(prices["SPY"].pct_change())
        mult = mult.reindex(rets.index).ffill()
        with open(cache_path, "wb") as f:
            pickle.dump(mult, f)

    nav = spy_nav = bh_nav = capital
    weights = pd.Series(1 / len(tickers), index=tickers)

    month_ends = set(rets.groupby([rets.index.year, rets.index.month]).tail(1).index)
    daily_hedge_drag = hedge_drag_annual / TRADING_DAYS
    spy_rets = prices["SPY"].pct_change().reindex(rets.index)

    curves = {"strategy": [], "spy": [], "equal_weight_bh": []}

    for date in rets.index:
        daily_ret = (rets.loc[date] * weights).sum()
        nav *= (1 + daily_ret - daily_hedge_drag)
        spy_nav *= 1 + spy_rets.loc[date]
        bh_nav *= 1 + rets.loc[date].mean()
        curves["strategy"].append(nav)
        curves["spy"].append(spy_nav)
        curves["equal_weight_bh"].append(bh_nav)

        if date in month_ends:
            window = rets.loc[:date].iloc[-126:]
            vols = window.std() * np.sqrt(TRADING_DAYS)
            win_rates = (window > 0).mean()
            avg_win = window.where(window > 0, np.nan).mean()
            avg_loss = abs(window.where(window <= 0, np.nan).mean())
            payoffs = (avg_win / avg_loss.replace(0, np.nan)).fillna(1.0)

            raw = (win_rates - (1 - win_rates) / payoffs).clip(lower=0) * 0.5
            reg_mult = mult.loc[date] if date in mult.index else 1.0
            targeted = raw * reg_mult * (0.15 / vols).clip(upper=1.0)
            weights = (targeted / targeted.sum() if targeted.sum() > 0
                       else pd.Series(1 / len(tickers), index=tickers))

    curve = pd.DataFrame(curves, index=rets.index)

    def metrics(series):
        r = series.pct_change().dropna()
        total = series.iloc[-1] / series.iloc[0] - 1
        years = len(r) / TRADING_DAYS
        ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
        dd = ((series - series.cummax()) / series.cummax()).min()
        sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() else 0
        sortino_denom = r[r < 0].std() * np.sqrt(TRADING_DAYS)
        sortino = r.mean() * TRADING_DAYS / sortino_denom if sortino_denom else 0
        return {"total_return": round(total, 4), "cagr": round(ann, 4),
                "max_drawdown": round(dd, 4), "sharpe": round(sharpe, 2),
                "sortino": round(sortino, 2)}

    return {"curve": curve,
            "strategy_metrics": metrics(curve["strategy"]),
            "spy_metrics": metrics(curve["spy"]),
            "bh_metrics": metrics(curve["equal_weight_bh"])}


def print_backtest_report(result: dict):
    print("\n⏪ STRATEGY BACKTEST (regime-aware Kelly loop vs benchmarks)")
    for name, label in [("strategy_metrics", "🧭 RegimeGuard"),
                        ("bh_metrics", "⚖️ Equal-weight B&H"),
                        ("spy_metrics", "📈 SPY")]:
        m = result[name]
        print(f"   {label:<18} CAGR {m['cagr']:+.1%} | MaxDD {m['max_drawdown']:.1%}"
              f" | Sharpe {m['sharpe']:.2f} | Sortino {m['sortino']:.2f}")

    curve = result["curve"]
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        (curve / curve.iloc[0]).plot(ax=ax)
        ax.set_title("RegimeGuard Strategy vs Benchmarks")
        ax.set_ylabel("Growth of $1")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("backtest.png", dpi=120)
        print("\n   📊 Equity curve saved to backtest.png")
    except ImportError:
        pass
