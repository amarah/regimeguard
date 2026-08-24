#!/usr/bin/env python3
"""RegimeGuard CLI.

Usage:
    python cli.py --tickers AAPL MSFT NVDA --capital 50000 --hedge --backtest
"""
import argparse
import json

import pandas as pd

from regimeguard.regimes import detect_regime
from regimeguard.sizing import size_positions
from regimeguard.risk import monte_carlo_cvar
from regimeguard.stress import run_stress_tests
from regimeguard.clusters import correlation_clusters
from regimeguard.alerts import notify_all
from regimeguard.options import (optimize_hedge, print_hedge_report,
                                 optimize_collar, print_collar_report,
                                 build_put_spreads, print_spread_report)
from regimeguard.skew import print_skew_report
from regimeguard.backtest import run_backtest, print_backtest_report


def main():
    ap = argparse.ArgumentParser(prog="regimeguard")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--capital", type=float, default=100_000)
    ap.add_argument("--notify", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--hedge", action="store_true", help="protective puts")
    ap.add_argument("--collar", action="store_true", help="zero-cost collars")
    ap.add_argument("--spreads", action="store_true", help="bear put spreads")
    ap.add_argument("--skew", action="store_true", help="IV skew analysis")
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()

    _, regime = detect_regime()
    sizes = size_positions(args.tickers, args.capital, regime.position_multiplier)
    total = sizes["dollars"].sum() if not sizes.empty else 0
    weights = {}
    if total > 0:
        weights = dict(zip(sizes["ticker"], sizes["dollars"] / total))

    risk = monte_carlo_cvar(weights) if weights else {}
    stress = run_stress_tests(weights) if weights else pd.DataFrame()
    clusters = correlation_clusters(args.tickers)

    if args.json:
        print(json.dumps({
            "date": str(pd.Timestamp.now().date()),
            "regime": regime.label,
            "confidence": regime.probability,
            "exposure_multiplier": regime.position_multiplier,
            "positions": sizes.to_dict("records"),
            "risk": risk,
        }, indent=2))
        return

    print("═" * 62)
    print(f"  REGIMEGUARD v1.0 — {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    print("═" * 62)
    print(f"\n🧭 REGIME: {regime.label} ({regime.probability:.0%} confidence)")
    print(f"   Trend {regime.ann_trend:+.1%}/yr · Vol {regime.ann_vol:.1%} · Exposure guide {regime.position_multiplier:.0%}")

    print("\n📊 POSITION SIZING (Kelly × regime × vol-target)")
    if not sizes.empty:
        print(sizes.to_string(index=False))
        print(f"\n   Total deployed: ${total:,.0f}")

    if risk:
        print("\n🎲 RISK (10-day horizon, fat-tailed MC)")
        print(f"   VaR  95% : {risk['VaR_95']:.2%}  (~${total*risk['VaR_95']:,.0f})")
        print(f"   CVaR 95% : {risk['CVaR_95']:.2%}  (~${total*risk['CVaR_95']:,.0f})")
        print(f"   Max DD at risk : {risk['DaR_95']:.2%}")
        print(f"   P(loss over 10d): {risk['prob_loss']:.0%}")

    if not stress.empty:
        print("\n💥 HISTORICAL CRISIS REPLAY")
        print(stress.to_string(index=False))

    print("\n🔗 EXPOSURE CLUSTERS")
    for g in clusters:
        flag = "⚠️ REDUNDANT" if len(g) > 1 else "✓ unique"
        print(f"   {' + '.join(g)} → {flag}")

    pos_vals = dict(zip(sizes["ticker"], sizes["dollars"])) if not sizes.empty else {}
    cvar = risk.get("CVaR_95", 0.05)

    if args.hedge and pos_vals:
        print_hedge_report(optimize_hedge(args.tickers, pos_vals, cvar), cvar)
    if args.collar and pos_vals:
        print_collar_report(optimize_collar(args.tickers, pos_vals))
    if args.spreads and pos_vals:
        print_spread_report(build_put_spreads(args.tickers, pos_vals))
    if args.skew:
        print_skew_report(args.tickers)
    if args.backtest:
        print_backtest_report(run_backtest(args.tickers, args.capital))

    if args.notify:
        msg = (f"*RegimeGuard* 🧭\nRegime: *{regime.label}* ({regime.probability:.0%})\n"
               f"Exposure: {regime.position_multiplier:.0%}\n"
               f"CVaR(10d): {risk.get('CVaR_95', 0):.2%}")
        channels = notify_all(msg)
        print(f"\n📨 Alert sent via: {', '.join(channels) or 'none configured'}")

    print("\n" + "═" * 62)
    print("⚠ Educational risk analysis only — not investment advice.")


if __name__ == "__main__":
    main()
