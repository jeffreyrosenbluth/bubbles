from typing import NamedTuple

import numpy as np
import polars as pl

from bubbles.dz import dz


class MarketParameters(NamedTuple):
    years: int = 50
    initial_expected_return: float = 0.04
    earnings_vol: float = 0.10
    payout_ratio: float = 0.5
    history_length: int = 5
    seed: int = 987654321  # 0 for dz.py

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
    ws = [start_weight * (0.75**i) for i in range(n)]
    s = sum(ws)
    return np.array([w / s for w in ws])


def weighted_avg_returns(return_idx: np.ndarray, weights: np.ndarray, t: int) -> float:
    indices = [t - i * 12 - 1 for i in range(len(weights) + 1)]
    returns_slice = return_idx[indices]
    result = 0
    for i in range(len(weights)):
        result += weights[i] * (returns_slice[i] / returns_slice[i + 1] - 1)
    return result


class InvestorParameters(NamedTuple):
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
            "None"
            if self.speed_of_adjustment is None
            else str(self.speed_of_adjustment)
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


class TimeSeries(NamedTuple):
    monthly_earnings: np.ndarray
    price_idx: np.ndarray
    annualized_earnings: np.ndarray
    return_idx: np.ndarray
    total_cash: np.ndarray
    expected_return_x: np.ndarray
    expected_return_y: np.ndarray
    squeeze: np.ndarray
    wealth_x: np.ndarray
    wealth_y: np.ndarray
    equity_x: np.ndarray
    equity_y: np.ndarray
    cash_x: np.ndarray
    cash_y: np.ndarray
    cash_post_distribution_x: np.ndarray
    cash_post_distribution_y: np.ndarray
    n_year_annualized_return: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    dz: np.ndarray


def initialize_time_series(
    m: MarketParameters,
) -> TimeSeries:
    n = 12 * (m.years + m.history_length) + 1
    return TimeSeries(
        monthly_earnings=np.full(n, np.nan, dtype=float),
        price_idx=np.ones(n),
        annualized_earnings=np.full(n, np.nan, dtype=float),
        return_idx=np.ones(n),
        total_cash=np.full(n, np.nan, dtype=float),
        expected_return_x=np.full(n, np.nan, dtype=float),
        expected_return_y=np.full(n, np.nan, dtype=float),
        squeeze=np.full(n, np.nan, dtype=float),
        wealth_x=np.full(n, np.nan, dtype=float),
        wealth_y=np.full(n, np.nan, dtype=float),
        equity_x=np.full(n, np.nan, dtype=float),
        equity_y=np.full(n, np.nan, dtype=float),
        cash_x=np.full(n, np.nan, dtype=float),
        cash_y=np.full(n, np.nan, dtype=float),
        cash_post_distribution_x=np.full(n, np.nan, dtype=float),
        cash_post_distribution_y=np.full(n, np.nan, dtype=float),
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
        pl.DataFrame with all TimeSeries arrays as columns
    """
    return pl.DataFrame(
        {
            "annualized_earnings": ts.annualized_earnings,
            "monthly_earnings": ts.monthly_earnings,
            "return_idx": ts.return_idx,
            "price_idx": ts.price_idx,
            "total_cash": ts.total_cash,
            "expected_return_x": ts.expected_return_x,
            "expected_return_y": ts.expected_return_y,
            "squeeze": ts.squeeze,
            "wealth_x": ts.wealth_x,
            "wealth_y": ts.wealth_y,
            "equity_x": ts.equity_x,
            "equity_y": ts.equity_y,
            "cash_x": ts.cash_x,
            "cash_y": ts.cash_y,
            "cash_post_distribution_x": ts.cash_post_distribution_x,
            "cash_post_distribution_y": ts.cash_post_distribution_y,
            "n_year_annualized_return": ts.n_year_annualized_return,
            "a": ts.a,
            "b": ts.b,
            "c": ts.c,
            "dz": ts.dz,
        }
    )
