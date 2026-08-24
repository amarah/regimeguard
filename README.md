# RegimeGuard

**Regime-aware risk and position sizing engine for systematic trading analysis.**

RegimeGuard goes beyond a simple entry signal. It analyzes the current market environment, estimates position sizes, quantifies portfolio tail risk, and identifies holdings that may be redundant exposures.

## What it does

1. **Market regime detection** fits a 3-state model to SPY returns, realized volatility, and quarterly momentum, labeling the current environment **RISK-ON**, **CHOPPY**, or **RISK-OFF**.
2. **Fractional Kelly sizing** estimates recent win rate and payoff, applies half-Kelly sizing, scales exposure by the detected regime, and applies volatility targeting.
3. **Monte Carlo CVaR** simulates correlated portfolio returns and reports 95% VaR, 95% CVaR / expected shortfall, worst simulated loss, and median return.
4. **Correlation clustering** — flags holdings with absolute return correlation ≥ 0.75 and groups them as potentially redundant exposures.

The implementation uses `hmmlearn` when available for the regime model and falls back to a Gaussian Mixture Model when it is not.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

```bash
python regimeguard.py \\
  --tickers AAPL MSFT NVDA TLT GLD \\
  --capital 50000 \\
  --period 3y
```

Arguments:

- `--tickers` — one or more ticker symbols; required.
- `--capital` — portfolio capital used for sizing; defaults to `100000`.
- `--period` — historical period used for regime detection; defaults to `5y`.

The engine fetches market data through Yahoo Finance via `yfinance`, so an internet connection is required when running it.

## Architecture

```text
SPY data
   │
   ├── returns + realized vol + momentum
   │              │
   │              ▼
   │       HMM / GMM regime model
   │              │
   │              ▼
   │       regime multiplier
   │              │
   ▼              ▼
Ticker history → Kelly + volatility targeting → position sizes
                                      │
                                      ▼
                              portfolio weights
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  Monte Carlo CVaR          correlation clusters
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                                risk summary
```

## Important assumptions

The Kelly estimates are based on recent daily return statistics and should not be interpreted as a forecast of future trade outcomes.

## Roadmap

Potential extensions include:

- historical crisis stress tests (2008, COVID, 2022)
- drawdown-at-risk analysis
- a Streamlit dashboard with regime probability charts
- scheduled GitHub Actions runs
- automated alerts
