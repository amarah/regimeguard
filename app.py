"""RegimeGuard Dashboard — run with: streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from regimeguard.data import get_prices
from regimeguard.regimes import detect_regime
from regimeguard.sizing import size_positions
from regimeguard.risk import monte_carlo_cvar
from regimeguard.stress import run_stress_tests
from regimeguard.clusters import correlation_clusters

st.set_page_config(page_title="RegimeGuard", page_icon="🧭", layout="wide")
st.title("🧭 RegimeGuard — Market Regime & Risk Engine")

with st.sidebar:
    tickers = st.text_input("Tickers", value="AAPL MSFT NVDA TLT GLD").split()
    capital = st.number_input("Capital ($)", value=50_000, step=1_000)
    sims = st.slider("MC simulations", 5_000, 50_000, 20_000, step=5_000)
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

if not run_btn:
    st.info("Configure your portfolio in the sidebar and hit ▶ Run Analysis.")
    st.stop()


@st.cache_data(show_spinner="Detecting regime...")
def cached_regime():
    _, r = detect_regime()
    return r


@st.cache_data(show_spinner="Sizing positions...")
def cached_sizes(_tickers, _capital, _mult):
    return size_positions(list(_tickers), _capital, _mult)


regime = cached_regime()
color = {"RISK-ON": "green", "CHOPPY": "orange", "RISK-OFF": "red"}[regime.label]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Regime", regime.label, f"{regime.probability:.0%} conf.")
c2.metric("Regime Trend", f"{regime.ann_trend:+.1%}", "/year")
c3.metric("Regime Volatility", f"{regime.ann_vol:.1%}", "annualized")
c4.metric("Exposure Guide", f"{regime.position_multiplier:.0%}", "of normal")

prices = get_prices(tickers + ["SPY"], "5y")
hist = pd.DataFrame({
    "SPY": prices["SPY"],
    "regime_code": regime.history.map({"RISK-OFF": 0, "CHOPPY": 1, "RISK-ON": 2}).reindex(prices.index),
})
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist["SPY"], name="SPY", line=dict(color="black", width=1)))
fig.add_trace(go.Scatter(x=hist.index, y=hist["regime_code"] * hist["SPY"].max(), fill="tozeroy", mode="none", name="Regime", line=dict(color=color, width=0), opacity=0.15))
fig.update_layout(title="SPY with Regime Shading", height=380)
st.plotly_chart(fig, use_container_width=True)

sizes = cached_sizes(tuple(tickers), capital, regime.position_multiplier)
lc, rc = st.columns([1.2, 1])

with lc:
    st.subheader("📊 Position Sizing")
    if sizes.empty:
        st.warning("No valid data for these tickers.")
    else:
        st.dataframe(sizes, use_container_width=True, hide_index=True)
        fig_pie = go.Figure(go.Pie(labels=sizes["ticker"], values=sizes["dollars"], hole=0.55))
        fig_pie.update_layout(title="Allocation", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

with rc:
    st.subheader("🎲 Risk Metrics")
    total = sizes["dollars"].sum() if not sizes.empty else 0
    if total > 0:
        weights = dict(zip(sizes["ticker"], sizes["dollars"] / total))
        risk = monte_carlo_cvar(weights, sims=sims)
        st.metric("10-day CVaR (95%)", f"{risk['CVaR_95']:.2%}", delta=f"-${total*risk['CVaR_95']:,.0f}", delta_color="inverse")
        st.metric("Max Drawdown at Risk", f"{risk['DaR_95']:.2%}")
        st.metric("P(loss in 10 days)", f"{risk['prob_loss']:.0%}")
        st.caption(f"Fat-tailed Student-t simulation · {sims:,} paths")

        st.subheader("💥 Crisis Replay")
        st.dataframe(run_stress_tests(weights), use_container_width=True, hide_index=True)

st.subheader("🔗 Correlation Clusters (ρ ≥ 0.75)")
for g in correlation_clusters(tickers):
    badge = "⚠️" if len(g) > 1 else "✅"
    st.markdown(f"{badge} `{'` + `'.join(g)}`")

st.divider()
st.caption("⚠️ Educational tool — not investment advice. Backtest before trading.")
