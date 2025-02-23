import textwrap
from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
import polars as pl

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


def weights_5_36(start_weight: float = 36.0, n: int = 5) -> np.ndarray:
    """Generate exponentially decaying weights for return calculations.

    Args:
        start_weight: Initial weight value
        n: Number of weights to generate

    Returns:
        Normalized array of weights that sum to 1.0
    """
    ws = [start_weight * (0.75**i) for i in range(n)]
    s = sum(ws)
    return np.array([w / s for w in ws])


def weighted_avg_returns(return_idx: np.ndarray, weights: np.ndarray, t: int) -> float:
    """Calculate weighted average of historical returns.

    Args:
        return_idx: Array of return indices
        weights: Array of weights for each historical period
        t: Current time index

    Returns:
        Weighted average of returns over the specified period
    """
    indices = [t - i * 12 - 1 for i in range(len(weights) + 1)]
    returns_slice = return_idx[indices]
    result = 0
    for i in range(len(weights)):
        result += weights[i] * (returns_slice[i] / returns_slice[i + 1] - 1)
    return result


class InvestorProvider(Protocol):
    """Protocol defining the interface for investor types.

    Methods:
        get_percent: Returns the investor's percentage of total market
        get_gamma: Returns the investor's risk aversion parameter
        get_sigma: Returns the investor's volatility parameter
        merton_share: Calculates optimal portfolio share based on Merton's formula
        get_expected_return: Returns array of expected returns over time
        get_wealth: Returns array of total wealth over time
        get_equity: Returns array of equity holdings over time
        get_cash: Returns array of cash holdings over time
        get_cash_post_distribution: Returns array of cash positions after distributions
    """

    def get_percent(self) -> float: ...
    def get_gamma(self) -> float: ...
    def get_sigma(self) -> float: ...

    def merton_share(self, excess_return: float) -> float:
        return excess_return / (self.get_gamma() * self.get_sigma() ** 2)

    def get_expected_return(self) -> np.ndarray[float]: ...
    def get_wealth(self) -> np.ndarray[float]: ...
    def get_equity(self) -> np.ndarray[float]: ...
    def get_cash(self) -> np.ndarray[float]: ...
    def get_cash_post_distribution(self) -> np.ndarray[float]: ...


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

    wealth: np.ndarray[float]
    expected_return: np.ndarray[float]
    equity: np.ndarray[float]
    cash: np.ndarray[float]
    cash_post_distribution: np.ndarray[float]

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
        def array_stats(arr: np.ndarray) -> str:
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


class Extrapolator(NamedTuple):
    """An investor type that extrapolates returns from historical data.

    Attributes:
        params: Basic investor parameters
        stats: Time series statistics for this investor
        weights: Array of weights for historical return calculation
        speed_of_adjustment: Rate at which expectations adjust to new information
        squeeze_target: Target return rate for normalization
        max_deviation: Maximum allowed deviation from squeeze target
        squeezing: Scaling factor for return normalization
    """

    params: InvestorParameters
    stats: InvestorStats
    weights: np.ndarray
    speed_of_adjustment: float
    squeeze_target: float
    max_deviation: float
    squeezing: float

    @classmethod
    def create(cls) -> "Extrapolator":
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

        def safe_mean(arr: np.ndarray) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "N/A"
            return f"{np.nanmean(arr[valid]):.2f}"

        def safe_mean_pct(arr: np.ndarray) -> str:
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

    def get_percent(self) -> float:
        return self.params.percent

    def get_gamma(self) -> float:
        return self.params.gamma

    def get_sigma(self) -> float:
        return self.params.sigma

    def get_wealth(self) -> np.ndarray[float]:
        return self.stats.wealth

    def get_expected_return(self) -> np.ndarray[float]:
        return self.stats.expected_return

    def get_equity(self) -> np.ndarray[float]:
        return self.stats.equity

    def get_cash(self) -> np.ndarray[float]:
        return self.stats.cash

    def get_cash_post_distribution(self) -> np.ndarray[float]:
        return self.stats.cash_post_distribution

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

    def merton_share(self, excess_return: float) -> float:
        return InvestorProvider.merton_share(self, excess_return)


class LongTermInvestor(NamedTuple):
    """An investor type that maintains constant return expectations.

    Attributes:
        params: Basic investor parameters
        stats: Time series statistics for this investor
    """

    params: InvestorParameters
    stats: InvestorStats

    @classmethod
    def create(cls) -> "LongTermInvestor":
        return cls(params=InvestorParameters(), stats=InvestorStats.initialize(Market()))

    def __repr__(self) -> str:
        def safe_mean(arr: np.ndarray) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "N/A"
            return f"{np.nanmean(arr[valid]):.2f}"

        def safe_mean_pct(arr: np.ndarray) -> str:
            valid = ~np.isnan(arr)
            if not np.any(valid):
                return "N/A"
            return f"{np.nanmean(arr[valid]):.2%}"

        return f"LongTermInvestor\n---------------\n{textwrap.indent(repr(self.params), '  ')}\n"

    def get_percent(self) -> float:
        return self.params.percent

    def get_gamma(self) -> float:
        return self.params.gamma

    def get_sigma(self) -> float:
        return self.params.sigma

    def get_wealth(self) -> np.ndarray[float]:
        return self.stats.wealth

    def get_expected_return(self) -> np.ndarray[float]:
        return self.stats.expected_return

    def get_equity(self) -> np.ndarray[float]:
        return self.stats.equity

    def get_cash(self) -> np.ndarray[float]:
        return self.stats.cash

    def get_cash_post_distribution(self) -> np.ndarray[float]:
        return self.stats.cash_post_distribution

    def merton_share(self, excess_return: float) -> float:
        return InvestorProvider.merton_share(self, excess_return)


class TimeSeries(NamedTuple):
    """Container for all time-varying simulation data.

    Attributes:
        monthly_earnings: Monthly earnings values
        price_idx: Price index over time
        annualized_earnings: Annualized earnings
        return_idx: Return index
        total_cash: Total cash in system
        investors: List of investors
        squeeze: Squeeze metric over time
        n_year_annualized_return: N-year annualized returns
        a, b, c: Intermediate calculation values
        dz: Random shock values
    """

    monthly_earnings: np.ndarray
    price_idx: np.ndarray
    annualized_earnings: np.ndarray
    return_idx: np.ndarray
    total_cash: np.ndarray
    investors: list[InvestorProvider]
    squeeze: np.ndarray
    n_year_annualized_return: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    dz: np.ndarray

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
            squeeze=np.full(n, np.nan, dtype=float),
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
                    f"expected_return{suffix}": investor.get_expected_return(),
                    f"wealth{suffix}": investor.get_wealth(),
                    f"equity{suffix}": investor.get_equity(),
                    f"cash{suffix}": investor.get_cash(),
                    f"cash_post_distribution{suffix}": investor.get_cash_post_distribution(),
                }
            )

        return pl.DataFrame(data)

    def __repr__(self) -> str:
        def array_stats(arr: np.ndarray) -> str:
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
