# 🧭 RegimeGuard

**A regime-aware portfolio risk engine for equity traders.**

RegimeGuard is designed to answer six practical questions: what market regime are we in, how much should each position be sized, how bad can the portfolio get, which holdings are secretly the same trade, how would the portfolio have behaved through major crashes, and what options hedges can address modeled tail risk.

> ⚠️ Educational risk analysis only — not investment advice.

## Features

| Module | What it does |
|---|---|
| Regime detection | 3-state HMM on SPY returns, realized volatility, and momentum; GMM fallback |
| Position sizing | Fractional Kelly × regime multiplier × volatility targeting |
| Tail risk | Fat-tailed Student-t Monte Carlo, VaR, CVaR / expected shortfall, Drawdown-at-Risk |
| Crisis replay | Historical portfolio stress tests for GFC, COVID, 2022 bear market, and Volmageddon |
| Correlation clusters | Union-find grouping of holdings with absolute return correlation ≥ 0.75 |
| Protective puts | CVaR-linked put candidate ranking |
| Collars | Zero-cost / low-cost protective collar finder |
| Put spreads | Bear put-spread builder with capped protection |
| IV skew | Downside-versus-upside implied-volatility analysis |
| Backtester | Monthly-rebalanced regime-aware strategy versus equal-weight buy-and-hold and SPY |
| Dashboard | Streamlit interface for regime, sizing, risk, stress, and clusters |
| Alerts | Discord and Telegram notifications via GitHub Actions |

## Repository structure

```text
regimeguard/
├── regimeguard/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── regimes.py
│   ├── sizing.py
│   ├── risk.py
│   ├── stress.py
│   ├── clusters.py
│   ├── options.py
│   ├── skew.py
│   ├── backtest.py
│   └── alerts.py
├── app.py
├── cli.py
├── .github/workflows/
│   ├── daily-regime.yml
│   └── ci.yml
├── tests/
│   └── test_risk.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .env.example
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the CLI:

```bash
python cli.py --tickers AAPL MSFT NVDA --capital 50000
```

Run the extended analysis:

```bash
python cli.py --tickers AAPL MSFT NVDA --capital 50000 --hedge --skew --backtest
```

JSON output is available with `--json`.

## Dashboard

```bash
streamlit run app.py
```

Configure tickers, capital, and Monte Carlo simulations in the sidebar, then run the analysis. The dashboard displays the current regime, regime statistics, SPY regime history, position sizing, tail-risk metrics, crisis replay, and correlation clusters.

## Docker

```bash
docker build -t regimeguard .
docker run --rm -p 8501:8501 regimeguard
```

Then open the Streamlit application on port 8501.

## Alerts and GitHub Actions

The scheduled workflow runs after the U.S. market close on weekdays and can send a compact regime/CVaR report through Discord and/or Telegram.

Add these GitHub Actions secrets:

| Secret | Purpose |
|---|---|
| `RG_DISCORD_WEBHOOK` | Discord incoming webhook |
| `RG_TELEGRAM_TOKEN` | Telegram bot token |
| `RG_TELEGRAM_CHAT_ID` | Telegram destination chat ID |

No secrets are stored in source code. `.env` is ignored locally and `.env.example` documents the expected environment variables.

## How the engine works

### 1. Market regime

SPY daily returns are combined with 21-day annualized realized volatility and 63-day momentum. A three-state Gaussian HMM is preferred; when `hmmlearn` is unavailable, the implementation falls back to a Gaussian Mixture Model. States are ranked by mean return and labeled **RISK-ON**, **CHOPPY**, and **RISK-OFF**.

The current state maps to exposure guidance of 100%, 60%, or 25% of normal respectively.

### 2. Position sizing

For each holding, recent daily returns estimate win rate and average win/loss payoff. Half-Kelly sizing is then multiplied by the regime exposure multiplier and reduced when annualized volatility exceeds the portfolio volatility target. Individual weights are capped by configuration.

### 3. Tail risk

Portfolio return paths are simulated with a correlated covariance structure. The default tail model uses Student-t scaling with five degrees of freedom. The engine reports 95% VaR, 95% CVaR, worst simulated loss, probability of loss, and Drawdown-at-Risk.

### 4. Historical stress

Actual portfolio weights are replayed through the configured historical crisis windows and compared with SPY. The report includes total portfolio return, maximum drawdown, SPY return, relative performance, and a simple recovery-time estimate.

### 5. Redundant exposures

Absolute daily-return correlations at or above 0.75 are joined with a union-find structure. This catches clusters where several tickers may effectively represent one portfolio bet.

### 6. Options hedging

The options module uses live Yahoo Finance option chains to rank protective puts, find low-cost collars, and build bear put spreads. The put optimizer links candidate protection to modeled CVaR. The skew module compares approximately 7%-out-of-the-money put and call implied volatility as a simple insurance-pricing proxy.

### 7. Backtesting

The backtester rebalances monthly using Kelly and volatility-targeting inputs and applies the regime multiplier. It compares the resulting equity curve with equal-weight buy-and-hold and SPY. The current implementation is explicitly a fast full-history regime fit; strict walk-forward refitting remains a roadmap item.

## Example CLI output

A typical run contains sections for:

```text
REGIMEGUARD v1.0

🧭 REGIME: RISK-ON (... confidence)
📊 POSITION SIZING (Kelly × regime × vol-target)
🎲 RISK (10-day horizon, fat-tailed MC)
💥 HISTORICAL CRISIS REPLAY
🔗 EXPOSURE CLUSTERS
🛡️ PROTECTIVE PUT OPTIMIZER
📐 IV SKEW ANALYSIS
⏪ STRATEGY BACKTEST
```

Exact values depend on the live market data returned by Yahoo Finance.

## Development and testing

Install the development dependencies from `requirements.txt`, then run:

```bash
pytest tests/ -v
```

The included tests cover Kelly bounds, Monte Carlo CVaR ordering, Black-Scholes put sanity bounds, and correlation-cluster membership. Tests that exercise market-data modules require network access to Yahoo Finance.

## Data and operational notes

- Market data is fetched through `yfinance`.
- Local price data is cached as Parquet for 12 hours by the package data layer.
- Option chains are live external data and can occasionally fail when Yahoo changes its API.
- Weekend runs can contain stale market quotes.
- A first long-period backtest can take materially longer because regime fitting is computationally heavier.
- The default scheduled workflow uses a fixed example portfolio; edit `.github/workflows/daily-regime.yml` for a different portfolio.

## Implemented vs roadmap

### Implemented

- [x] 3-state HMM / GMM regime engine
- [x] Fractional Kelly sizing
- [x] Volatility targeting
- [x] Student-t Monte Carlo CVaR
- [x] Drawdown-at-Risk
- [x] Historical crisis replay
- [x] Correlation clustering
- [x] Protective put optimizer
- [x] Collar optimizer
- [x] Bear put-spread builder
- [x] IV skew analyzer
- [x] Strategy backtester
- [x] Streamlit dashboard
- [x] Discord / Telegram alerts
- [x] GitHub Actions CI and scheduled analysis
- [x] Docker support

### Roadmap

- [ ] Strict walk-forward regime refitting in the backtester
- [ ] More sophisticated option-chain liquidity and spread-cost filters
- [ ] Additional historical stress regimes and custom user-defined scenarios
- [ ] Portfolio-level optimization across hedge instruments
- [ ] Persistent dashboard deployment and richer regime visualizations

## Disclaimer

RegimeGuard is an educational quantitative-risk analysis project. Its outputs are estimates based on historical market data, model assumptions, and simulated scenarios. They are not guarantees of future performance and are not investment, trading, legal, or financial advice.
