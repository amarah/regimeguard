"""IV skew analysis — is downside insurance cheap or rich today?"""
import yfinance as yf


def analyze_skew(ticker: str) -> dict:
    """
    Compare today's ~25%-moneyness put/call IV spread.
    Very negative skew = expensive crash insurance; unusually flat = cheap.
    """
    tk = yf.Ticker(ticker)
    try:
        spot = tk.fast_info.get("last_price")
    except Exception:
        return {}
    if not spot:
        return {}
    expiry = next((e for e in tk.options
                   if 20 <= (pd.Timestamp(e) - pd.Timestamp.now()).days <= 45), None)
    if not expiry:
        return {}

    puts = tk.option_chain(expiry).puts
    calls = tk.option_chain(expiry).calls

    def iv_near_moneyness(df, side):
        target = spot * (0.93 if side == "put" else 1.07)
        sub = df[df["strike"] < target] if side == "put" else df[df["strike"] > target]
        if sub.empty:
            return None
        idx = (sub["strike"] - target).abs().idxmin()
        return float(sub.loc[idx, "impliedVolatility"])

    put_iv = iv_near_moneyness(puts, "put")
    call_iv = iv_near_moneyness(calls, "call")
    if not put_iv or not call_iv:
        return {}
    skew = put_iv - call_iv

    verdict = ("⚠️ Insurance EXPENSIVE — market pricing crash fear"
               if skew > 0.06 else
               "✅ Insurance reasonably CHEAP — good time to hedge"
               if skew < 0.02 else
               "➖ Skew near normal")

    return {"ticker": ticker, "expiry": expiry,
            "put_iv_25d_proxy": round(put_iv, 4),
            "call_iv_25d_proxy": round(call_iv, 4),
            "skew": round(skew, 4), "verdict": verdict}


def print_skew_report(tickers: list[str]):
    print("\n📐 IV SKEW ANALYSIS (downside insurance pricing)")
    for t in tickers:
        r = analyze_skew(t)
        if r:
            print(f"   {t:<6} put IV {r['put_iv_25d_proxy']:.1%} | "
                  f"call IV {r['call_iv_25d_proxy']:.1%} | "
                  f"skew {r['skew']:+.3f} → {r['verdict']}")
