"""Monte Carlo VaR/CVaR and Drawdown-at-Risk using fat-tailed Student-t."""
import numpy as np

from .data import get_prices


def monte_carlo_cvar(weights: dict[str, float], horizon_days: int = 10,
                     sims: int = 20_000, alpha: float = 0.95,
                     student_t: bool = True) -> dict:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not isinstance(horizon_days, int) or isinstance(horizon_days, bool) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer.")
    if not isinstance(sims, int) or isinstance(sims, bool) or sims <= 0:
        raise ValueError("sims must be a positive integer.")
    if not weights:
        raise ValueError("At least one portfolio weight is required.")
    try:
        supplied_weights = np.asarray(list(weights.values()), dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Portfolio weights must be finite numbers.") from None
    if not np.isfinite(supplied_weights).all():
        raise ValueError("Portfolio weights must be finite numbers.")
    if np.isclose(supplied_weights.sum(), 0):
        raise ValueError("Portfolio weights must have a nonzero net total.")
    tickers = list(weights.keys())
    rets = get_prices(tickers, period="3y").pct_change(fill_method=None).dropna()
    if len(rets) < 2:
        raise ValueError("Risk estimation needs at least two aligned return observations.")
    mu, cov = rets.mean().values, rets.cov().values
    w = np.array([float(weights.get(c, 0)) for c in rets.columns])
    if np.isclose(w.sum(), 0):
        raise ValueError("Downloaded assets must have a nonzero net portfolio weight.")
    w /= w.sum()

    rng = np.random.default_rng(7)
    L = np.linalg.cholesky(cov + np.eye(len(mu)) * 1e-12)
    z = rng.standard_normal((sims, horizon_days, len(mu)))

    if student_t:
        nu = 5
        scale = np.sqrt(nu / rng.chisquare(nu, size=(sims, horizon_days, 1)))
        # A raw Student-t draw has variance nu / (nu - 2). Standardize it so
        # the fitted covariance still controls volatility while tails get heavier.
        z *= scale * np.sqrt((nu - 2) / nu)

    paths = (z @ L.T + mu) @ w
    cumulative = np.prod(1 + paths, axis=1) - 1

    var = np.percentile(cumulative, (1 - alpha) * 100)
    cvar = cumulative[cumulative <= var].mean()

    equity = np.cumprod(1 + paths, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    dd = ((equity - running_max) / running_max).min(axis=1)
    dd_var = np.percentile(dd, (1 - alpha) * 100)

    confidence = f"{alpha * 100:.6f}".rstrip("0").rstrip(".").replace(".", "_")
    return {
        f"VaR_{confidence}": round(float(-var), 4),
        f"CVaR_{confidence}": round(float(-cvar), 4),
        "worst_sim": round(float(-cumulative.min()), 4),
        "median_return": round(float(np.median(cumulative)), 4),
        "prob_loss": round(float((cumulative < 0).mean()), 4),
        f"DaR_{confidence}": round(float(-dd_var), 4),
    }
