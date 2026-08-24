"""
Options overlay module:
  1. Protective put optimizer (CVaR-linked)
  2. Zero-cost collar optimizer
  3. Bear put-spread builder
All use real option chains via yfinance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PutCandidate:
    ticker: str
    expiry: str
    strike: float
    spot: float
    premium: float
    cost_pct: float
    protection_floor: str
    days_to_expiry: int
    efficiency: float


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes_put(S: float, K: float, T: float, r: float = 0.04,
                      sigma: float = 0.25) -> float:
    """Theoretical put price — sanity-checks market quotes."""
    if T <= 0 or S <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return max(K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1), 0)


def fetch_put_chain(ticker: str, min_dte: int = 20, max_dte: int = 60) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    try:
        spot = tk.fast_info.get("last_price") or tk.history(period="1d")["Close"].iloc[-1]
    except Exception:
        return pd.DataFrame()
    rows = []
    for expiry in tk.options:
        dte = (pd.Timestamp(expiry) - pd.Timestamp.now()).days
        if not (min_dte <= dte <= max_dte):
            continue
        chain = tk.option_chain(expiry).puts
        otm = chain[(chain["strike"] >= spot * 0.80) &
                    (chain["strike"] <= spot * 0.995)].copy()
        otm["expiry"], otm["dte"], otm["spot"] = expiry, dte, spot
        rows.append(otm)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows)


def optimize_hedge(tickers: list[str], position_values: dict[str, float],
                   cvar_pct: float, top_n: int = 5,
                   target_coverage: float = 0.75) -> pd.DataFrame:
    results = []
    for t in tickers:
        pos_val = position_values.get(t, 0)
        if pos_val <= 0:
            continue
        chain = fetch_put_chain(t)
        if chain.empty:
            print(f"⚠ No suitable put chain found for {t}")
            continue
        for _, row in chain.iterrows():
            strike, prem, spot = row["strike"], row["lastPrice"], row["spot"]
            if prem <= 0 or np.isnan(prem):
                continue
            contracts_needed = int(pos_val // (strike * 100)) or 1
            total_cost = contracts_needed * prem * 100
            cost_pct = total_cost / pos_val
            floor_loss_pct = 1 - (strike / spot)
            efficiency = floor_loss_pct / max(cost_pct, 1e-6)
            results.append(PutCandidate(
                ticker=t, expiry=row["expiry"], strike=round(strike, 2),
                spot=round(spot, 2), premium=round(prem, 2),
                cost_pct=round(cost_pct, 4),
                protection_floor=f"{-floor_loss_pct:+.1%}",
                days_to_expiry=int(row["dte"]),
                efficiency=round(efficiency, 2)))
    df = pd.DataFrame([vars(c) for c in results])
    if df.empty:
        return df
    df = df.sort_values("efficiency", ascending=False).head(top_n)
    needed_protection = cvar_pct * target_coverage
    df["covers_cvar_target"] = df["protection_floor"].str.rstrip("%").astype(float) / 100 <= -needed_protection
    return df.reset_index(drop=True)


def print_hedge_report(df: pd.DataFrame, cvar_pct: float):
    print("\n🛡️ PROTECTIVE PUT OPTIMIZER")
    print(f"   Your modeled tail loss (CVaR): {cvar_pct:.2%}")
    print("   Target coverage: 75% of tail loss\n")
    if df.empty:
        print("   No candidates found.")
        return
    cols = ["ticker", "expiry", "strike", "spot", "premium",
            "cost_pct", "protection_floor", "days_to_expiry", "efficiency"]
    print(df[cols].to_string(index=False))
    for _, r in df.iterrows():
        mark = "✓" if r["covers_cvar_target"] else "✗"
        print(f"   {mark} {r['ticker']} {r['strike']:.0f}P exp {r['expiry']} @ ${r['premium']:.2f} ({r['cost_pct']:.1%} drag)")


def optimize_collar(tickers: list[str], position_values: dict[str, float],
                    max_cost_pct: float = 0.01,
                    min_floor: float = -0.20) -> pd.DataFrame:
    """Zero-cost collar finder: buy protective put funded by covered call."""
    rows = []
    for t in tickers:
        pos_val = position_values.get(t, 0)
        if pos_val <= 0:
            continue
        tk = yf.Ticker(t)
        try:
            spot = tk.fast_info.get("last_price")
        except Exception:
            continue
        if not spot:
            continue
        expiry = next((e for e in tk.options if 30 <= (pd.Timestamp(e) - pd.Timestamp.now()).days <= 60), None)
        if not expiry:
            continue
        dte = (pd.Timestamp(expiry) - pd.Timestamp.now()).days
        puts = tk.option_chain(expiry).puts
        calls = tk.option_chain(expiry).calls

        cand_puts = puts[(puts["strike"] >= spot * (1 + min_floor)) &
                         (puts["strike"] <= spot * 0.92)]
        cand_calls = calls[(calls["strike"] > spot * 1.05) &
                           (calls["strike"] < spot * 1.25)]

        for _, p in cand_puts.iterrows():
            if p["lastPrice"] <= 0:
                continue
            best_call, best_diff = None, float("inf")
            for _, c in cand_calls.iterrows():
                diff = abs(c["bid"] - p["ask"])
                if diff < best_diff:
                    best_diff, best_call = diff, c
            if best_call is None or best_call["bid"] <= 0:
                continue
            n_contracts = max(int(pos_val // (p["strike"] * 100)), 1)
            net_cost = (p["ask"] - best_call["bid"]) * 100 * n_contracts
            cost_pct = net_cost / pos_val
            if cost_pct > max_cost_pct:
                continue
            rows.append({
                "ticker": t, "expiry": expiry, "dte": dte,
                "spot": round(spot, 2),
                "buy_put": round(p["strike"], 2),
                "sell_call": round(best_call["strike"], 2),
                "put_ask": round(p["ask"], 2),
                "call_bid": round(best_call["bid"], 2),
                "net_cost_per_set": round(p["ask"] - best_call["bid"], 2),
                "total_net_cost": round(net_cost, 2),
                "cost_pct": round(cost_pct, 4),
                "floor": f"{(p['strike']/spot-1):+.0%}",
                "cap": f"{(best_call['strike']/spot-1):+.0%}",
                "contracts": n_contracts})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["cost_pct", "floor"], ascending=[True, False]).head(10)


def print_collar_report(df: pd.DataFrame):
    print("\n🛡️💰 COLLAR OPTIMIZER (put bought / call sold per contract set)")
    if df.empty:
        print("   No viable zero-cost collars found.")
        return
    print(df[["ticker", "expiry", "buy_put", "sell_call", "net_cost_per_set",
              "cost_pct", "floor", "cap"]].to_string(index=False))
    print("\n   Trade-off: protection below 'floor' paid for by giving up gains above 'cap'.")


def build_put_spreads(tickers: list[str], position_values: dict[str, float],
                      width_targets: tuple = (0.05, 0.10, 0.15)) -> pd.DataFrame:
    """Bear put spreads: cheaper capped protection than naked puts."""
    rows = []
    for t in tickers:
        pos_val = position_values.get(t, 0)
        if pos_val <= 0:
            continue
        tk = yf.Ticker(t)
        try:
            spot = tk.fast_info.get("last_price")
        except Exception:
            continue
        if not spot:
            continue
        expiry = next((e for e in tk.options if 30 <= (pd.Timestamp(e) - pd.Timestamp.now()).days <= 60), None)
        if not expiry:
            continue
        chain = tk.option_chain(expiry).puts.set_index("strike")

        for w in width_targets:
            long_k = round(spot * 0.95 / 0.5) * 0.5
            short_k = round(long_k * (1 - w) / 0.5) * 0.5
            if long_k not in chain.index or short_k not in chain.index:
                continue
            debit = chain.loc[long_k, "ask"] - chain.loc[short_k, "bid"]
            if debit <= 0:
                continue
            max_profit = (long_k - short_k) - debit
            contracts = max(int(pos_val // (long_k * 100)), 1)
            rows.append({
                "ticker": t, "expiry": expiry,
                "spread": f"{long_k:.0f}/{short_k:.0f}P",
                "width_pct": w, "debit": round(debit, 2),
                "max_gain_ps": round(max_profit, 2),
                "roi_on_debit": round(max_profit / debit, 2),
                "total_cost": round(debit * 100 * contracts, 2),
                "contracts": contracts,
                "protection_if_below_short_k": f"{max(w, (long_k-short_k-debit)/spot):.1%}"})
    return pd.DataFrame(rows)


def print_spread_report(df: pd.DataFrame):
    print("\n📉 PUT SPREAD BUILDER (bear verticals)")
    if df.empty:
        print("   No valid spreads found.")
        return
    print(df.to_string(index=False))
    print("\n   Cheaper than naked puts; protection capped at spread width.")
