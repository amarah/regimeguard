"""Zero-volatility prices agree with the positive-volatility limit."""
import math

import pytest

from regimeguard.options import black_scholes_put


@pytest.mark.parametrize("spot,strike,rate", [(90, 100, 0.04), (100, 100, -0.02), (99, 100, 0.04)])
def test_zero_volatility_discounts_strike(spot, strike, rate):
    expected = max(strike * math.exp(-rate) - spot, 0)
    assert black_scholes_put(spot, strike, 1, rate, 0) == pytest.approx(expected)
    assert black_scholes_put(spot, strike, 1, rate, 1e-8) == pytest.approx(expected)


def test_expiry_still_uses_intrinsic_value():
    assert black_scholes_put(90, 100, 0, 0.04, 0) == 10


@pytest.mark.parametrize(
    "values,message",
    [
        ((float("nan"), 100, 1, 0.04, 0.2), "finite numbers"),
        ((100, 0, 1, 0.04, 0.2), "positive"),
        ((100, 100, -1, 0.04, 0.2), "cannot be negative"),
        ((100, 100, 1, 0.04, -0.2), "cannot be negative"),
        ((100, 100, 100, -100, 0.2), "supported range"),
    ],
)
def test_invalid_model_inputs_are_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        black_scholes_put(*values)
