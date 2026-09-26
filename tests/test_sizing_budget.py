"""Portfolio budget regression tests using fixed price data."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from regimeguard.sizing import size_positions


class SizingBudgetTests(unittest.TestCase):
    def size(self, count, capital=10000.0, weight=0.25):
        tickers = [f"T{i}" for i in range(count)]
        prices = pd.DataFrame({ticker: [100.0, 100.1, 100.15, 100.3]
                               for ticker in tickers})
        with patch("regimeguard.sizing.get_prices", return_value=prices), \
                patch("regimeguard.sizing.kelly_fraction", return_value=weight):
            return size_positions(tickers, capital, 1.0)

    def test_five_positions_cannot_exceed_capital(self):
        positions = self.size(5)
        self.assertLessEqual(positions.dollars.sum(), 10000.0)
        np.testing.assert_allclose(positions.weight, 0.2)
        np.testing.assert_allclose(positions.dollars, 2000.0)

    def test_unused_cash_is_not_forced_into_positions(self):
        positions = self.size(2, weight=0.1)
        np.testing.assert_allclose(positions.weight, 0.1)
        np.testing.assert_allclose(positions.dollars, 1000.0)

    def test_cent_rounding_does_not_overallocate(self):
        positions = self.size(6, capital=100.0)
        self.assertLessEqual(positions.dollars.sum(), 100.0)
        np.testing.assert_allclose(positions.dollars, 16.66)

    def test_share_counts_use_scaled_budget(self):
        positions = self.size(5)
        self.assertTrue((positions.shares == 19).all())
        self.assertLessEqual((positions.shares * positions.price).sum(), 10000.0)

    def test_relative_allocations_are_preserved(self):
        tickers = [f"T{i}" for i in range(6)]
        prices = pd.DataFrame({ticker: [100, 100.1, 100.15, 100.3] for ticker in tickers})
        weights = [0.25, 0.25, 0.25, 0.15, 0.15, 0.15]
        with patch("regimeguard.sizing.get_prices", return_value=prices), \
                patch("regimeguard.sizing.kelly_fraction", side_effect=weights):
            positions = size_positions(tickers, 12000.0, 1.0)
        np.testing.assert_allclose(positions.dollars, [2500, 2500, 2500, 1500, 1500, 1500], atol=0.011)
        self.assertLessEqual(positions.dollars.sum(), 12000.0)

    def test_empty_price_series_remains_empty(self):
        with patch("regimeguard.sizing.get_prices", return_value=pd.DataFrame({"T": []})):
            self.assertTrue(size_positions(["T"], 10000, 1).empty)


if __name__ == "__main__":
    unittest.main()
