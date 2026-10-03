"""Offline tests for option-overlay sizing."""

import pytest

from regimeguard.options import contracts_for_position


def test_contracts_cover_the_entire_share_position():
    assert contracts_for_position(10_000, 100) == 1
    assert contracts_for_position(10_000.01, 100) == 2
    assert contracts_for_position(25_000, 100) == 3


def test_contract_sizing_handles_empty_and_invalid_positions():
    assert contracts_for_position(0, 100) == 0
    assert contracts_for_position(-1, 100) == 0
    with pytest.raises(ValueError, match="Spot price"):
        contracts_for_position(1_000, 0)
