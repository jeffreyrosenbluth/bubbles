from __future__ import annotations

import textwrap
from typing import NamedTuple

import numpy as np
import polars as pl
from numpy.typing import NDArray

from bubbles.dz import dz
from bubbles.market import Market  # Updated import
from bubbles.protocols import InvestorProvider

SQRT_12 = np.sqrt(12)


def weights_5_36(start_weight: float = 36.0, n: int = 5) -> NDArray[np.float64]:
    """Generate exponentially decaying weights for return calculations.

    Args:
        start_weight: Initial weight value
        n: Number of weights to generate

    Returns:
        Normalized array of weights that sum to 1.0
    """
    ws = start_weight * np.power(0.75, np.arange(n))  # Vectorized exponentiation
    return ws / ws.sum()


def weighted_avg_returns(
    return_idx: NDArray[np.float64], weights: NDArray[np.float64], t: int
) -> float:
    """Calculate weighted average of historical returns.

    Args:
        return_idx: Array of return indices
        weights: Array of weights for each historical period
        t: Current time index

    Returns:
        Weighted average of returns over the specified period
    """
    indices = t - np.arange(len(weights) + 1) * 12 - 1
    returns_slice = return_idx[indices]
    return np.sum(weights * (returns_slice[:-1] / returns_slice[1:] - 1))


class TimeSeries(NamedTuple):
    """Container for all time-varying simulation data.

    Attributes:
        monthly_earnings: Monthly earnings values
        price_idx: Price index over time
        annualized_earnings: Annualized earnings
        return_idx: Return index
        total_cash: Total cash in system
        investors: List of investors
        n_year_annualized_return: N-year annualized returns
        a, b, c: Intermediate calculation values
        dz: Random shock values
    """

    monthly_earnings: NDArray[np.float64]
    price_idx: NDArray[np.float64]
    annualized_earnings: NDArray[np.float64]
    return_idx: NDArray[np.float64]
    total_cash: NDArray[np.float64]
    investors: list[InvestorProvider]
    n_year_annualized_return: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    c: NDArray[np.float64]
    dz: NDArray[np.float64]

    @classmethod
    def initialize(cls, mkt: Market, investors: list[InvestorProvider]) -> "TimeSeries":
        """Create a new TimeSeries instance with initialized arrays.

        Args:
            m: Market parameters defining simulation length

        Returns:
            New TimeSeries instance with appropriately sized arrays
        """
        n = 12 * (mkt.years + mkt.history_length) + 1
        dz_zero = dz.copy()
        dz_zero[: mkt.history_length * 12 + 1] = 0
        return cls(
            monthly_earnings=np.full(n, np.nan, dtype=float),
            price_idx=np.ones(n),  # Initialize to ones
            annualized_earnings=np.full(n, np.nan, dtype=float),
            return_idx=np.ones(n),  # Initialize to ones
            total_cash=np.full(n, np.nan, dtype=float),
            investors=investors,  # Initialize with empty list
            n_year_annualized_return=np.full(n, np.nan, dtype=float),
            a=np.full(n, np.nan, dtype=float),
            b=np.full(n, np.nan, dtype=float),
            c=np.full(n, np.nan, dtype=float),
            dz=dz_zero,
        )

    def earnings(self, mkt: Market, reinvested: float, t: int) -> float:
        """Calculate earnings for the current period.

        Args:
            mkt: Market parameters
            reinvested: Amount of earnings reinvested
            t: Current time index

        Returns:
            Earnings value for current period
        """
        return (
            self.monthly_earnings[t - 1]
            * (1 + reinvested / self.price_idx[t - 1])
            * (1 + self.dz[t] * mkt.earnings_vol / SQRT_12)
        )

    def annualize(self, mkt: Market, t: int) -> float:
        """Convert monthly earnings to annualized value.

        Args:
            mkt: Market parameters
            t: Current time index

        Returns:
            Annualized earnings value
        """
        return ((1 + self.monthly_earnings[t] / self.price_idx[t - 1]) ** 12 - 1) * self.price_idx[
            t - 1
        ]

    def calculate_return_idx(self, mkt: Market, t: int) -> float:
        """Calculate return index including both price changes and dividends.

        Args:
            mkt: Market parameters
            t: Current time index

        Returns:
            Updated return index value
        """
        return self.return_idx[t - 1] * (
            (self.price_idx[t] + self.monthly_earnings[t] * mkt.payout_ratio)
            / self.price_idx[t - 1]
        )

    def to_df(self) -> pl.DataFrame:
        """Convert this TimeSeries instance to a Polars DataFrame.

        Returns:
            Polars DataFrame with all TimeSeries arrays as columns
        """
        data = {
            "annualized_earnings": self.annualized_earnings,
            "monthly_earnings": self.monthly_earnings,
            "return_idx": self.return_idx,
            "price_idx": self.price_idx,
            "total_cash": self.total_cash,
            "squeeze": self.squeeze,
            "n_year_annualized_return": self.n_year_annualized_return,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "dz": self.dz,
        }

        # Add data for each investor
        for i, investor in enumerate(self.investors):
            suffix = f"_{i}"
            data.update(
                {
                    f"expected_return{suffix}": investor.expected_return(),
                    f"wealth{suffix}": investor.wealth(),
                    f"equity{suffix}": investor.equity(),
                    f"cash{suffix}": investor.cash(),
                    f"cash_post_distribution{suffix}": investor.cash_post_distribution(),
                }
            )

        return pl.DataFrame(data)

    def __repr__(self) -> str:
        def array_stats(arr: NDArray[np.float64]) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "no valid data"
            data = arr[valid]
            return f"mean: {data.mean():.2f}, min: {data.min():.2f}, max: {data.max():.2f}"

        n = len(self.price_idx)
        investors_str = "\n".join(
            f"  Investor {i}:\n{textwrap.indent(repr(inv), '    ')}"
            for i, inv in enumerate(self.investors, 1)
        )

        return (
            f"TimeSeries (length: {n})\n"
            f"---------------------\n"
            f"Market Data:\n"
            f"  monthly_earnings: {array_stats(self.monthly_earnings)}\n"
            f"  price_idx: {array_stats(self.price_idx)}\n"
            f"  annualized_earnings: {array_stats(self.annualized_earnings)}\n"
            f"  return_idx: {array_stats(self.return_idx)}\n"
            f"  total_cash: {array_stats(self.total_cash)}\n"
            f"  squeeze: {array_stats(self.squeeze)}\n"
            f"  n_year_annualized_return: {array_stats(self.n_year_annualized_return)}\n\n"
            f"Calculation Values:\n"
            f"  a: {array_stats(self.a)}\n"
            f"  b: {array_stats(self.b)}\n"
            f"  c: {array_stats(self.c)}\n"
            f"  dz: {array_stats(self.dz)}\n\n"
            f"Investors:\n"
            f"{investors_str}\n"
        )
