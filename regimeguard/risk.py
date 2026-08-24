"""Monte Carlo VaR/CVaR and Drawdown-at-Risk using fat-tailed Student-t."""
import numpy as np

from .data import get_prices


def monte_carlo_cvar(weights: dict[str, float], horizon_days: int = 10,
                     sims: int = 20_000, alpha: float = 0.95,
                     student_t: bool = True) -> dict:
    tickers = list(weights.keys())
    rets = get_prices(tickers, period="3y").pct_change().dropna()
    mu, cov = rets.mean().values, rets.cov().values
    w = np.array([weights.get(c, 0) for c in rets.columns])
    w /= w.sum()

    rng = np.random.default_rng(7)
    L = np.linalg.cholesky(cov + np.eye(len(mu)) * 1e-12)
    z = rng.standard_normal((sims, horizon_days, len(mu)))

    if student_t:
        nu = 5
        scale = np.sqrt(nu / rng.chisquare(nu, size=(sims, horizon_days, 1)))
        z *= scale

    paths = (z @ L.T + mu) @ w
    cumulative = np.prod(1 + paths, axis=1) - 1

    var = np.percentile(cumulative, (1 - alpha) * 100)
    cvar = cumulative[cumulative <= var].mean()

    equity = np.cumprod(1 + paths, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    dd = ((equity - running_max) / running_max).min(axis=1)
    dd_var = np.percentile(dd, (1 - alpha) * 100)

    return {
        "VaR_95": round(float(-var), 4),
        "CVaR_95": round(float(-cvar), 4),
        "worst_sim": round(float(-cumulative.min()), 4),
        "median_return": round(float(np.median(cumulative)), 4),
        "prob_loss": round(float((cumulative < 0).mean()), 4),
        "DaR_95": round(float(-dd_var), 4),
    }
