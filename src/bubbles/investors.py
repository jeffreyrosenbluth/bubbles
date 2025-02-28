import textwrap
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from bubbles.core import Market


@dataclass(frozen=True)
class InvestorParameters:
    """Parameters defining an investor's characteristics.

    Attributes:
        percent: Investor's percentage of total market
        gamma: Risk aversion parameter
        sigma: Volatility parameter
    """

    percent: float
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

    def equity(self) -> NDArray[np.float64]:
        return self.stats.equity

    def cash(self) -> NDArray[np.float64]:
        return self.stats.cash

    def expected_return(self) -> NDArray[np.float64]:
        return self.stats.expected_return

    def cash_post_distribution(self) -> NDArray[np.float64]:
        return self.stats.cash_post_distribution

    def merton_share(self, excess_return: np.float64) -> np.float64:
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
    def new(cls, percent: float) -> "Extrapolator":
        return cls(
            params=InvestorParameters(percent),
            stats=InvestorStats.initialize(Market()),
            weights=cls.weights_5_36(),
            speed_of_adjustment=0.1,
            squeeze_target=0.04,
            max_deviation=0.04,
            squeezing=0.1,
        )

    def investor_type(self) -> Literal["extrapolator"]:
        return "extrapolator"

    @staticmethod
    def weights_5_36(start_weight: float = 36.0, n: int = 5) -> NDArray[np.float64]:
        """Generate exponentially decaying weights for return calculations.

        Args:
            start_weight: Initial weight value
            n: Number of weights to generate

        Returns:
            Normalized array of weights that sum to 1.0
        """
        ws = start_weight * np.power(0.75, np.arange(n))
        return np.array(ws / ws.sum(), dtype=np.float64)

    def weighted_avg_returns(
        self,
        return_idx: NDArray[np.float64],
        weights: NDArray[np.float64],
        t: int,
    ) -> np.float64:
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
        return np.float64(np.sum(weights * (returns_slice[:-1] / returns_slice[1:] - 1)))

    def calculate_expected_return(
        self,
        t: int,
        n_year_annualized_return: float,
        mkt: Market,
    ) -> float:
        squeeze = self.squeeze_target + self.max_deviation * np.tanh(
            (n_year_annualized_return - mkt.initial_expected_return) / self.squeezing
        )
        return float(
            squeeze * self.speed_of_adjustment
            + (1 - self.speed_of_adjustment) * self.expected_return()[t - 1]
        )

    def desired_equity(
        self,
        t: int,
        n_year_annualized_return: np.float64,
        mkt: Market,
        price_prev: np.float64,
        price_new: np.float64,
    ) -> np.float64:
        er = self.calculate_expected_return(t, n_year_annualized_return, mkt)
        return (
            er
            * (self.cash_post_distribution()[t] + price_new * self.equity()[t - 1] / price_prev)
            / (self.gamma() * self.sigma() ** 2)
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


@dataclass
class LongTermInvestor(InvestorBase):
    """An investor type that maintains constant return expectations.

    Attributes:
        params: Basic investor parameters
        stats: Time series statistics for this investor
    """

    @classmethod
    def new(cls, percent: float) -> "LongTermInvestor":
        return cls(params=InvestorParameters(percent), stats=InvestorStats.initialize(Market()))

    def investor_type(self) -> Literal["long_term"]:
        return "long_term"

    def calculate_expected_return(
        self,
        annualized_earnings: np.float64,
        price: np.float64,
    ) -> np.float64:
        return annualized_earnings / price

    def desired_equity(
        self,
        t: int,
        annualized_earnings: np.float64,
        price_prev: np.float64,
        price_new: np.float64,
    ) -> np.float64:
        er = self.calculate_expected_return(annualized_earnings, price_new)
        return (
            er
            * (self.cash_post_distribution()[t] + price_new * self.equity()[t - 1] / price_prev)
            / (self.gamma() * self.sigma() ** 2)
        )

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


@dataclass
class Rebalancer_60_40(InvestorBase):
    """An investor type that rebalances between stocks and bonds.

    Attributes:
        params: Basic investor parameters
        stats: Time series statistics for this investor
    """

    @classmethod
    def new(cls, percent: float) -> "Rebalancer_60_40":
        return cls(params=InvestorParameters(percent), stats=InvestorStats.initialize(Market()))

    def investor_type(self) -> Literal["rebalancer_60_40"]:
        return "rebalancer_60_40"

    def calculate_expected_return(self) -> np.float64:
        return 0.0

    def desired_equity(self) -> np.float64:
        return 0.6

    def __repr__(self) -> str:
        return f"Rebalancer_60_40\n---------------\n{textwrap.indent(repr(self.params), '  ')}\n"
