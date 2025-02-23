import textwrap
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray

from bubbles.dz import dz

SQRT_12 = np.sqrt(12)


class Market(NamedTuple):
    """Global parameters for market simulation.

    Attributes:
        years: Number of years to simulate
        initial_expected_return: Starting expected return rate
        earnings_vol: Volatility of earnings
        payout_ratio: Ratio of earnings paid as dividends
        history_length: Number of years of history to consider
        seed: Random seed for reproducibility (0 uses predefined dz.py values)
    """

    years: int = 50
    initial_expected_return: float = 0.04
    earnings_vol: float = 0.10
    payout_ratio: float = 0.5
    history_length: int = 5
    seed: int = 1337194922  # 0 for dz.py

    def __repr__(self) -> str:
        return (
            f"Market Parameters\n"
            f"-----------------\n"
            f"  years: {self.years}\n"
            f"  initial expected_return: {self.initial_expected_return:.2%}\n"
            f"  earnings vol: {self.earnings_vol:.2%}\n"
            f"  payout ratio: {self.payout_ratio:.2%}\n"
            f"  history length: {self.history_length}\n"
            f"  seed: {self.seed}\n"
        )


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


class InvestorProvider(Protocol):
    """Protocol defining the interface for investor types.

    Methods:
        percent: Returns the investor's percentage of total market
        gamma: Returns the investor's risk aversion parameter
        sigma: Returns the investor's volatility parameter
        merton_share: Calculates optimal portfolio share based on Merton's formula
        expected_return: Returns array of expected returns over time
        wealth: Returns array of total wealth over time
        equity: Returns array of equity holdings over time
        cash: Returns array of cash holdings over time
        cash_post_distribution: Returns array of cash positions after distributions
    """

    def percent(self) -> float: ...
    def gamma(self) -> float: ...
    def sigma(self) -> float: ...

    def merton_share(self, excess_return: float) -> float:
        return excess_return / (self.gamma() * self.sigma() ** 2)

    def expected_return(self) -> NDArray[np.float64]: ...
    def wealth(self) -> NDArray[np.float64]: ...
    def equity(self) -> NDArray[np.float64]: ...
    def cash(self) -> NDArray[np.float64]: ...
    def cash_post_distribution(self) -> NDArray[np.float64]: ...


class InvestorParameters(NamedTuple):
    """Parameters defining an investor's characteristics.

    Attributes:
        percent: Investor's percentage of total market
        gamma: Risk aversion parameter
        sigma: Volatility parameter
    """

    percent: float = 0.5
    gamma: float = 3.0
    sigma: float = 0.16

    def __repr__(self) -> str:
        return (
            f"Investor Parameters\n"
            f"-----------------\n"
            f"  percent: {self.percent:.2%}\n"
            f"  gamma: {self.gamma:.2f}\n"
            f"  sigma: {self.sigma:.2%}\n"
        )


@dataclass
class InvestorStats:
    """Container for investor-specific time series data.

    Attributes:
        wealth: Total wealth over time
        expected_return: Expected returns over time
        equity: Equity holdings over time
        cash: Cash holdings over time
        cash_post_distribution: Cash position after distributions
    """

    wealth: NDArray[np.float64]
    expected_return: NDArray[np.float64]
    equity: NDArray[np.float64]
    cash: NDArray[np.float64]
    cash_post_distribution: NDArray[np.float64]

    @classmethod
    def initialize(cls, m: Market) -> "InvestorStats":
        """Create a new InvestorStats instance with arrays initialized to NaN.

        Args:
            m: Market parameters defining simulation length

        Returns:
            New InvestorStats instance with appropriately sized arrays
        """
        n = 12 * (m.years + m.history_length) + 1
        return cls(
            wealth=np.full(n, np.nan, dtype=float),
            expected_return=np.full(n, np.nan, dtype=float),
            equity=np.full(n, np.nan, dtype=float),
            cash=np.full(n, np.nan, dtype=float),
            cash_post_distribution=np.full(n, np.nan, dtype=float),
        )

    def __repr__(self) -> str:
        def array_stats(arr: NDArray[np.float64]) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "no valid data"
            data = arr[valid]
            return f"mean: {data.mean():.2f}, min: {data.min():.2f}, max: {data.max():.2f}"

        return (
            f"Investor Stats (length: {len(self.wealth)})\n"
            f"------------------------------------\n"
            f"  wealth: {array_stats(self.wealth)}\n"
            f"  expected_return: {array_stats(self.expected_return)}\n"
            f"  equity: {array_stats(self.equity)}\n"
            f"  cash: {array_stats(self.cash)}\n"
            f"  cash_post_distribution: {array_stats(self.cash_post_distribution)}\n\n"
        )


@dataclass
class InvestorBase:
    """Base class implementing common getter methods for investors."""

    params: InvestorParameters
    stats: InvestorStats

    def percent(self) -> float:
        return self.params.percent

    def gamma(self) -> float:
        return self.params.gamma

    def sigma(self) -> float:
        return self.params.sigma

    def wealth(self) -> NDArray[np.float64]:
        return self.stats.wealth

    def expected_return(self) -> NDArray[np.float64]:
        return self.stats.expected_return

    def equity(self) -> NDArray[np.float64]:
        return self.stats.equity

    def cash(self) -> NDArray[np.float64]:
        return self.stats.cash

    def cash_post_distribution(self) -> NDArray[np.float64]:
        return self.stats.cash_post_distribution

    def merton_share(self, excess_return: float) -> float:
        return excess_return / (self.gamma() * self.sigma() ** 2)


@dataclass
class Extrapolator(InvestorBase):
    """An investor type that extrapolates returns from historical data.

    Attributes:
        weights: Array of weights for historical return calculation
        speed_of_adjustment: Rate at which expectations adjust to new information
        squeeze_target: Target return rate for normalization
        max_deviation: Maximum allowed deviation from squeeze target
        squeezing: Scaling factor for return normalization
    """

    weights: NDArray[np.float64]
    speed_of_adjustment: float
    squeeze_target: float
    max_deviation: float
    squeezing: float

    @classmethod
    def new(cls) -> "Extrapolator":
        return cls(
            params=InvestorParameters(),
            stats=InvestorStats.initialize(Market()),
            weights=weights_5_36(),
            speed_of_adjustment=0.1,
            squeeze_target=0.04,
            max_deviation=0.04,
            squeezing=0.1,
        )

    def __repr__(self) -> str:
        weights_str = np.array2string(self.weights, precision=3, separator=", ")

        def safe_mean(arr: NDArray[np.float64]) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "N/A"
            return f"{np.nanmean(arr[valid]):.2f}"

        def safe_mean_pct(arr: NDArray[np.float64]) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "N/A"
            return f"{np.nanmean(arr[valid]):.2%}"

        return (
            f"Extrapolator\n"
            f"-----------\n"
            f"Parameters:\n"
            f"  speed_of_adjustment: {self.speed_of_adjustment:.3f}\n"
            f"  squeeze_target: {self.squeeze_target:.2%}\n"
            f"  max_deviation: {self.max_deviation:.2%}\n"
            f"  squeezing: {self.squeezing:.3f}\n"
            f"  weights: {weights_str}\n\n"
            f"{textwrap.indent(repr(self.params), '  ')}\n"
        )

    def normalize_weights(
        self, n_year_annualized_return: float, initial_expected_return: float
    ) -> float:
        """Normalize returns using a hyperbolic tangent transformation.

        Args:
            n_year_annualized_return: The n-year annualized return rate
            initial_expected_return: The starting expected return rate

        Returns:
            Normalized return rate bounded by squeeze_target ± max_deviation
        """
        return self.squeeze_target + self.max_deviation * np.tanh(
            (n_year_annualized_return - initial_expected_return) / self.squeezing
        )


@dataclass
class LongTermInvestor(InvestorBase):
    """An investor type that maintains constant return expectations.

    Attributes:
        params: Basic investor parameters
        stats: Time series statistics for this investor
    """

    @classmethod
    def new(cls) -> "LongTermInvestor":
        return cls(params=InvestorParameters(), stats=InvestorStats.initialize(Market()))

    def __repr__(self) -> str:
        def safe_mean(arr: NDArray[np.float64]) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "N/A"
            return f"{np.nanmean(arr[valid]):.2f}"

        def safe_mean_pct(arr: NDArray[np.float64]) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "N/A"
            return f"{np.nanmean(arr[valid]):.2%}"

        return f"LongTermInvestor\n---------------\n{textwrap.indent(repr(self.params), '  ')}\n"


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
