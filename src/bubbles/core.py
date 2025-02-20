from typing import NamedTuple

import numpy as np
import polars as pl

from bubbles.dz import dz


class MarketParameters(NamedTuple):
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


def return_weights(start_weight: float = 36.0, n: int = 5) -> np.ndarray:
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


class InvestorParameters(NamedTuple):
    """Parameters defining investor behavior.

    Attributes:
        percent: Investment allocation percentage
        gamma: Risk aversion parameter
        sigma: Volatility parameter
        weights: Optional custom weights for return calculations
        speed_of_adjustment: Optional parameter for return expectation adjustments
    """

    percent: float = 0.5
    gamma: float = 3.0
    sigma: float = 0.16
    weights: np.ndarray | None = None  # field(default_factory=return_weights)
    speed_of_adjustment: float | None = None  # 0.1

    def __repr__(self) -> str:
        weights_str = "None"
        if self.weights is not None:
            if callable(self.weights):
                weights = self.weights()
                weights_str = np.array2string(weights, precision=3)
            else:
                weights_str = np.array2string(self.weights, precision=3)
        speed_of_adjustment_str = (
            "None" if self.speed_of_adjustment is None else str(self.speed_of_adjustment)
        )
        return (
            f"Investor Parameters\n"
            f"--------------------\n"
            f"  percent: {self.percent:.2%}\n"
            f"  gamma: {self.gamma:.2}\n"
            f"  volatility: {self.sigma:.2%}\n"
            f"  weights: {weights_str}\n"
            f"  speed of adjustment: {speed_of_adjustment_str}\n"
        )


class SqueezeParameters(NamedTuple):
    """Parameters controlling market squeeze behavior.

    Attributes:
        squeeze_target: Target return during squeeze periods
        max_deviation: Maximum allowed deviation from target
        squeezing: Intensity of the squeeze effect
    """

    squeeze_target: float = 0.04
    max_deviation: float = 0.04
    squeezing: float = 0.1

    def __repr__(self) -> str:
        return (
            f"Squeeze Parameters\n"
            f"-------------------\n"
            f"  squeeze target: {self.squeeze_target:.2%}\n"
            f"  max deviation: {self.max_deviation:.2%}\n"
            f"  squeezing: {self.squeezing:.2%}\n"
        )


class Investor(NamedTuple):
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


def initialize_investor(m: MarketParameters) -> Investor:
    """Create a new Investor instance with arrays initialized to NaN.

    Args:
        m: Market parameters defining simulation length

    Returns:
        New Investor instance with appropriately sized arrays
    """
    n = 12 * (m.years + m.history_length) + 1
    return Investor(
        wealth=np.full(n, np.nan, dtype=float),
        expected_return=np.full(n, np.nan, dtype=float),
        equity=np.full(n, np.nan, dtype=float),
        cash=np.full(n, np.nan, dtype=float),
        cash_post_distribution=np.full(n, np.nan, dtype=float),
    )


class TimeSeries(NamedTuple):
    """Container for all time-varying simulation data.

    Attributes:
        monthly_earnings: Monthly earnings values
        price_idx: Price index over time
        annualized_earnings: Annualized earnings
        return_idx: Return index
        total_cash: Total cash in system
        investor_x: First investor's data
        investor_y: Second investor's data
        squeeze: Squeeze metric over time
        n_year_annualized_return: N-year annualized returns
        a, b, c: Intermediate calculation values
        dz: Random shock values
    """

    monthly_earnings: np.ndarray[float]
    price_idx: np.ndarray[float]
    annualized_earnings: np.ndarray[float]
    return_idx: np.ndarray[float]
    total_cash: np.ndarray[float]
    investor_x: Investor
    investor_y: Investor
    squeeze: np.ndarray[float]
    n_year_annualized_return: np.ndarray[float]
    a: np.ndarray[float]
    b: np.ndarray[float]
    c: np.ndarray[float]
    dz: np.ndarray[float]


def initialize_time_series(
    m: MarketParameters,
) -> TimeSeries:
    """Create a new TimeSeries instance with initialized arrays.

    Args:
        m: Market parameters defining simulation length

    Returns:
        New TimeSeries instance with appropriately sized arrays
    """
    n = 12 * (m.years + m.history_length) + 1
    return TimeSeries(
        monthly_earnings=np.full(n, np.nan, dtype=float),
        price_idx=np.ones(n),
        annualized_earnings=np.full(n, np.nan, dtype=float),
        return_idx=np.ones(n),
        total_cash=np.full(n, np.nan, dtype=float),
        squeeze=np.full(n, np.nan, dtype=float),
        investor_x=initialize_investor(m),
        investor_y=initialize_investor(m),
        n_year_annualized_return=np.full(n, np.nan, dtype=float),
        a=np.full(n, np.nan, dtype=float),
        b=np.full(n, np.nan, dtype=float),
        c=np.full(n, np.nan, dtype=float),
        dz=dz.copy(),
    )


def to_df(ts: TimeSeries) -> pl.DataFrame:
    """Convert a TimeSeries instance to a Polars DataFrame.

    Args:
        ts: TimeSeries instance containing the simulation data

    Returns:
        Polars DataFrame with all TimeSeries arrays as columns
    """
    return pl.DataFrame(
        {
            "annualized_earnings": ts.annualized_earnings,
            "monthly_earnings": ts.monthly_earnings,
            "return_idx": ts.return_idx,
            "price_idx": ts.price_idx,
            "total_cash": ts.total_cash,
            "expected_return_x": ts.investor_x.expected_return,
            "expected_return_y": ts.investor_y.expected_return,
            "squeeze": ts.squeeze,
            "wealth_x": ts.investor_x.wealth,
            "wealth_y": ts.investor_y.wealth,
            "equity_x": ts.investor_x.equity,
            "equity_y": ts.investor_y.equity,
            "cash_x": ts.investor_x.cash,
            "cash_y": ts.investor_y.cash,
            "cash_post_distribution_x": ts.investor_x.cash_post_distribution,
            "cash_post_distribution_y": ts.investor_y.cash_post_distribution,
            "n_year_annualized_return": ts.n_year_annualized_return,
            "a": ts.a,
            "b": ts.b,
            "c": ts.c,
            "dz": ts.dz,
        }
    )
