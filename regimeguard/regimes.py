"""Market regime detection via HMM (preferred) or GMM fallback."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
from sklearn.mixture import GaussianMixture

from .config import TRADING_DAYS
from .data import get_prices


@dataclass
class RegimeState:
    label: str
    probability: float
    ann_vol: float
    ann_trend: float
    position_multiplier: float
    history: pd.Series


MULTIPLIERS = {"RISK-ON": 1.0, "CHOPPY": 0.6, "RISK-OFF": 0.25}


def detect_regime(period: str = "5y") -> tuple[pd.DataFrame, RegimeState]:
    spy = get_prices(["SPY"], period)["SPY"]
    rets = spy.pct_change().dropna()
    vol = rets.rolling(21).std() * np.sqrt(TRADING_DAYS)
    mom = spy.pct_change(63)
    X = pd.concat([rets, vol, mom], axis=1).dropna()

    if HAS_HMM:
        model = GaussianHMM(n_components=3, covariance_type="full",
                            n_iter=300, random_state=42)
        model.fit(X.values)
        states = model.predict(X.values)
        probs = model.predict_proba(X.values)[-1]
    else:
        model = GaussianMixture(n_components=3, covariance_type="full",\                                n_init=10, random_state=42)
        model.fit(X.values)
        states = model.predict(X.values)
        probs = model.predict_proba(X.values)[-1]

    tmp = X.copy()
    tmp["state"] = states
    means = tmp.groupby("state")["ret"].mean().sort_values(ascending=False)
    label_map = {means.index[0]: "RISK-ON",
                 means.index[1]: "CHOPPY",
                 means.index[2]: "RISK-OFF"}
    labels = pd.Series(states, index=X.index).map(label_map)

    cur = int(states[-1])
    sub = tmp[tmp["state"] == cur]
    state = RegimeState(
        label=label_map[cur],
        probability=round(float(probs[cur]), 3),
        ann_vol=round(float(sub["vol"].mean()), 4),
        ann_trend=round(float(sub["ret"].mean() * TRADING_DAYS), 4),
        position_multiplier=MULTIPLIERS[label_map[cur]],
        history=labels,
    )
    return tmp, state
