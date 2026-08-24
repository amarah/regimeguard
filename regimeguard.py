"""
RegimeGuard — Regime-Aware Risk & Position Sizing Engine

Features:
  1. Hidden Markov Model regime detection on SPY (bull / bear / chop)
  2. Fractional-Kelly position sizing adjusted by current regime
  3. Monte Carlo CVaR (Expected Shortfall) for your portfolio
  4. Correlation clustering to detect redundant exposures

Usage:
    python regimeguard.py --tickers AAPL MSFT NVDA TLT GLD --capital 50000 --period 3y
"""

import argparse
import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.mixture import GaussianMixture

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

warnings.filterwarnings("ignore")

TRADING_DAYS = 252


# ============================================================
# 1. REGIME DETECTION
# ============================================================

@dataclass
class RegimeState:
    label: str            # "RISK-ON", "RISK-OFF", "CHOPPY"
    probability: float
    spy_ann_vol: float
    spy_trend: float      # annualized drift of current regime
    position_multiplier: float


def detect_regime(period: str = "5y") -> tuple[pd.DataFrame, RegimeState]:
    """Fit a 3-state HMM (or GMM) on SPY returns + realized vol."""
    spy = yf.download("SPY", period=period, auto_adjust=True)["Close"]
    rets = spy.pct_change().dropna()

    # Feature: rolling volatility + return
    vol = rets.rolling(21).std() * np.sqrt(TRADING_DAYS)
    mom = spy.pct_change(63)  # quarterly momentum
    X = pd.concat([rets, vol, mom], axis=1).dropna().values.reshape(-1, 3)

    if HAS_HMM:
        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)
        states = model.predict(X)
        probs = model.predict_proba(X)[-1]
    else:
        model = GaussianMixture(
            n_components=3,
            covariance_type="full",
            n_init=10,
            random_state=42,
        )
        model.fit(X)
        states = model.predict(X)
        probs = model.predict_proba(X)[-1]

    # Identify which state is which by mean return
    df = pd.DataFrame(X, columns=["ret", "vol", "mom"])
    df["state"] = states
    means = df.groupby("state")["ret"].mean()
    ranked = means.sort_values(ascending=False).index.tolist()
    labels = {
        ranked[0]: "RISK-ON",
        ranked[2]: "RISK-OFF",
        ranked[1]: "CHOPPY",
    }

    # Position multiplier: scale exposure by regime quality
    mult = {"RISK-ON": 1.0, "CHOPPY": 0.6, "RISK-OFF": 0.25}
    cur = int(states[-1])
    state_df = df[df["state"] == cur]
    ann_ret = state_df["ret"].mean() * TRADING_DAYS
    ann_vol = state_df["vol"].mean()

    info = RegimeState(
        label=labels[cur],
        probability=round(float(probs[cur]), 3),
        spy_ann_vol=round(float(ann_vol), 4),
        spy_trend=round(float(ann_ret), 4),
        position_multiplier=mult[labels[cur]],
    )
    df["spy"] = spy.reindex(df.index).values
    return df, info


# ============================================================
# 2. KELLY POSITION SIZING
# ============================================================

def kelly_fraction(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.5,
) -> float:
    """
    f* = p - (1-p)/b, scaled by `fraction` (half-Kelly default).
    Clamped to [0, 0.25] for sanity.
    """
    b = max(win_loss_ratio, 1e-9)
    f = win_rate - (1 - win_rate) / b
    return float(np.clip(fraction * f, 0.0, 0.25))


def estimate_trade_stats(prices: pd.Series, lookback: int = 126):
    """Estimate win rate & payoff from recent swing history."""
    rets = prices.pct_change().dropna().iloc[-lookback:]
    pos_days = rets > 0
    win_rate = float(pos_days.mean())
    avg_win = float(rets[pos_days].mean()) if pos_days.any() else 0.01
    avg_loss = abs(float(rets[~pos_days].mean())) if (~pos_days).any() else 0.01
    return win_rate, avg_win / avg_loss


def size_positions(
    tickers: list[str],
    capital: float,
    regime_mult: float,
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        px = yf.download(t, period="1y", auto_adjust=True)["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        wr, blr = estimate_trade_stats(px)
        raw_kelly = kelly_fraction(wr, blr, fraction=1.0)
        sized = kelly_fraction(wr, blr, fraction=0.5) * regime_mult

        # Volatility targeting: cap each position's vol contribution
        vol = float(px.pct_change().std() * np.sqrt(TRADING_DAYS))
        vol_target_weight = min(0.20 / vol, 1.0)  # 20% annual vol cap
        final_w = sized * vol_target_weight

        rows.append({
            "ticker": t,
            "price": round(float(px.iloc[-1]), 2),
            "win_rate": round(wr, 3),
            "payoff": round(blr, 2),
            "full_kelly": round(raw_kelly, 3),
            "regime_adj_half_kelly": round(final_w, 3),
            "dollars": round(capital * final_w, 2),
            "shares": int(capital * final_w // float(px.iloc[-1])),
        })
    return pd.DataFrame(rows)


# ============================================================
# 3. MONTE CARLO CVaR (EXPECTED SHORTFALL)
# ============================================================

def monte_carlo_cvar(
    weights: dict[str, float],
    horizon_days: int = 10,
    sims: int = 20_000,
    alpha: float = 0.95,
    period: str = "3y",
) -> dict:
    """Simulate correlated returns and report VaR & CVaR."""
    data = yf.download(
        list(weights.keys()),
        period=period,
        auto_adjust=True,
    )["Close"].pct_change().dropna()
    mu = data.mean().values
    cov = data.cov().values
    w = np.array([weights.get(c, 0) for c in data.columns])
    w = w / w.sum()

    rng = np.random.default_rng(7)
    # Cholesky-correlated multivariate normal draws
    L = np.linalg.cholesky(cov + np.eye(len(mu)) * 1e-12)
    z = rng.standard_normal((sims, horizon_days, len(mu)))
    paths = z @ L.T + mu
    port_paths = paths @ w
    cumulative = np.prod(1 + port_paths, axis=1) - 1

    var_alpha = np.percentile(cumulative, (1 - alpha) * 100)
    cvar_alpha = cumulative[cumulative <= var_alpha].mean()
    return {
        "horizon_days": horizon_days,
        "VaR_95": round(float(-var_alpha), 4),
        "CVaR_95": round(float(-cvar_alpha), 4),
        "worst_sim": round(float(-cumulative.min()), 4),
        "median_return": round(float(np.median(cumulative)), 4),
    }


# ============================================================
# 4. CORRELATION CLUSTERING (redundant-exposure detector)
# ============================================================

def correlation_clusters(
    tickers: list[str],
    period: str = "2y",
    threshold: float = 0.75,
) -> list[list[str]]:
    data = yf.download(tickers, period=period, auto_adjust=True)["Close"]
    corr = data.pct_change().dropna().corr()
    pairs = []
    for a, b in itertools.combinations(corr.columns, 2):
        r = corr.loc[a, b]
        if abs(r) >= threshold:
            pairs.append((a, b, r))

    # Union-find to group correlated names
    parent = {t: t for t in corr.columns}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups = {}
    for t in corr.columns:
        groups.setdefault(find(t), []).append(t)
    return list(groups.values())


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="RegimeGuard")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--period", default="5y")
    args = ap.parse_args()

    print("═" * 60)
    print("  REGIMEGUARD — Market Regime & Risk Engine")
    print("═" * 60)

    print("\n[1/4] Detecting market regime (HMM on SPY)...")
    _, regime = detect_regime(args.period)
    print(
        f"  Current regime : {regime.label} "
        f"(confidence {regime.probability:.0%})"
    )
    print(
        f"  Regime stats   : trend {regime.spy_trend:+.1%}/yr, "
        f"vol {regime.spy_ann_vol:.1%}"
    )
    print(f"  Exposure guide : {regime.position_multiplier:.0%} of normal")

    print("\n[2/4] Kelly-based position sizing...")
    sizes = size_positions(args.tickers, args.capital, regime.position_multiplier)
    print(sizes.to_string(index=False))
    total_alloc = sizes["dollars"].sum()
    print(
        f"\n  Total allocation : ${total_alloc:,.0f} "
        f"({total_alloc / args.capital:.0%} of capital)"
    )

    print("\n[3/4] Running Monte Carlo CVaR...")
    weights = dict(
        zip(
            sizes["ticker"],
            sizes["dollars"] / total_alloc if total_alloc else [0] * len(sizes),
        )
    )
    risk = monte_carlo_cvar(weights)
    print(
        f"  10-day VaR  (95%) : {risk['VaR_95']:.2%} "
        f"(≈ ${total_alloc * risk['VaR_95']:,.0f})"
    )
    print(
        f"  10-day CVaR (95%) : {risk['CVaR_95']:.2%} "
        f"(≈ ${total_alloc * risk['CVaR_95']:,.0f}) ← expected loss if breached"
    )
    print(f"  Worst simulation  : {risk['worst_sim']:.2%}")

    print("\n[4/4] Detecting redundant exposures (ρ ≥ 0.75)...")
    clusters = correlation_clusters(args.tickers)
    for group in clusters:
        flag = "⚠️  REDUNDANT" if len(group) > 1 else "✓ unique"
        print(f"  {' + '.join(group)}  → {flag}")
    if any(len(group) > 1 for group in clusters):
        print("  Tip: clustered names behave as one bet; consider diversifying.")

    print("\n" + "═" * 60)
    print("⚠ Educational risk analysis only — not investment advice.")


if __name__ == "__main__":
    main()
