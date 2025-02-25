from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from bubbles.core import InvestorProvider, Market
from bubbles.dz import dz

SQRT_12 = np.sqrt(12)


@dataclass(frozen=True)
class TimeSeries:
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
            monthly_earnings=np.full(n, np.nan, dtype=np.float64),
            price_idx=np.ones(n, dtype=np.float64),  # Initialize to ones
            annualized_earnings=np.full(n, np.nan, dtype=np.float64),
            return_idx=np.ones(n, dtype=np.float64),  # Initialize to ones
            total_cash=np.full(n, np.nan, dtype=np.float64),
            investors=investors,  # Initialize with empty list
            n_year_annualized_return=np.full(n, np.nan, dtype=np.float64),
            a=np.full(n, np.nan, dtype=np.float64),
            b=np.full(n, np.nan, dtype=np.float64),
            c=np.full(n, np.nan, dtype=np.float64),
            dz=dz_zero,
        )

    def earnings(self, mkt: Market, reinvested: np.float64, t: int) -> np.float64:
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

    def annualize(self, mkt: Market, t: int) -> np.float64:
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

    def calculate_return_idx(self, mkt: Market, t: int) -> np.float64:
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
            f"  n_year_annualized_return: {array_stats(self.n_year_annualized_return)}\n\n"
            f"Calculation Values:\n"
            f"  a: {array_stats(self.a)}\n"
            f"  b: {array_stats(self.b)}\n"
            f"  c: {array_stats(self.c)}\n"
            f"  dz: {array_stats(self.dz)}\n\n"
            f"Investors:\n"
            f"{investors_str}\n"
        )
