from dataclasses import dataclass, field

import numpy as np
import polars as pl

from bubbles.dz import dz

SQRT_12 = np.sqrt(12)


@dataclass
class Market:
    years: int = 50
    initial_expected_return: float = 0.04
    earnings_vol: float = 0.10
    payout_ratio: float = 0.5
    history_length: int = 5


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


@dataclass
class Investors:
    percent_y: float = 0.5
    percent_x: float = 1 - percent_y
    gamma_y: float = 3.0
    gamma_x: float = 3.0
    sigma_y: float = 0.16
    sigma_x: float = 0.16
    return_weights_x: np.ndarray = field(default_factory=return_weights)
    speed_of_adjustment: float = 0.1


@dataclass
class Squeeze:
    squeeze_target: float = 0.04
    max_deviation: float = 0.04
    squeezing: float = 0.1


def quadratic(a: float, b: float, c: float) -> float:
    discriminant = b**2 - 4 * a * c
    return (-b - np.sqrt(discriminant)) / (2 * a)


@dataclass
class TimeSeries:
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
    m: Market,
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


def history_ts(m: Market, investors: Investors) -> TimeSeries:
    s = initialize_time_series(m)

    s.monthly_earnings[1] = (1 + m.initial_expected_return) ** (1 / 12) - 1
    reinvested = s.monthly_earnings[1] * (1 - m.payout_ratio)
    s.price_idx[1] = s.price_idx[0] + reinvested
    s.annualized_earnings[1] = (
        (1 + s.monthly_earnings[1] / s.price_idx[0]) ** 12 - 1
    ) * s.price_idx[0]
    s.return_idx[1] = s.return_idx[0] * (
        (s.price_idx[1] + s.monthly_earnings[1] - reinvested) / s.price_idx[0]
    )
    s.dz[: m.history_length + 1] = 0  # Zero-out initial noise

    for t in range(2, m.history_length + 1):
        s.monthly_earnings[t] = (
            s.monthly_earnings[t - 1]
            * (1 + reinvested / s.price_idx[t - 1])
            * (1 + s.dz[t] * m.earnings_vol / SQRT_12)
        )
        reinvested = s.monthly_earnings[t] * (1 - m.payout_ratio)

        s.price_idx[t] = s.price_idx[t - 1] + reinvested
        s.annualized_earnings[t] = (
            (1 + s.monthly_earnings[t] / s.price_idx[t - 1]) ** 12 - 1
        ) * s.price_idx[t - 1]
        s.return_idx[t] = s.return_idx[t - 1] * (
            (s.price_idx[t] + s.monthly_earnings[t] - reinvested) / s.price_idx[t - 1]
        )

    s.expected_return_y[m.history_length] = (
        s.annualized_earnings[m.history_length] / s.price_idx[m.history_length]
    )
    s.n_year_annualized_return[m.history_length] = s.expected_return_y[m.history_length]

    # Merton shares
    merton_share_y = (
        s.annualized_earnings[m.history_length]
        / s.price_idx[m.history_length]
        / (investors.gamma_y * investors.sigma_y**2)
    )
    merton_share_x = (
        s.annualized_earnings[m.history_length]
        / s.price_idx[m.history_length]
        / (investors.gamma_x * investors.sigma_x**2)
    )

    starting_wealth = s.price_idx[m.history_length] / (
        investors.percent_y * merton_share_y + investors.percent_x * merton_share_x
    )

    s.total_cash[m.history_length] = starting_wealth - s.price_idx[m.history_length]
    s.expected_return_x[m.history_length] = s.expected_return_y[m.history_length]
    s.wealth_x[m.history_length] = investors.percent_x * starting_wealth
    s.wealth_y[m.history_length] = investors.percent_y * starting_wealth
    s.equity_x[m.history_length] = (
        s.wealth_x[m.history_length]
        * s.expected_return_x[m.history_length]
        / (investors.gamma_x * investors.sigma_x**2)
    )
    s.equity_y[m.history_length] = (
        s.wealth_y[m.history_length]
        * s.expected_return_y[m.history_length]
        / (investors.gamma_y * investors.sigma_y**2)
    )
    s.cash_x[m.history_length] = (
        s.wealth_x[m.history_length] - s.equity_x[m.history_length]
    )
    s.cash_y[m.history_length] = (
        s.wealth_y[m.history_length] - s.equity_y[m.history_length]
    )
    return s


def data_table(
    m: Market,
    investors: Investors,
    squeeze_params: Squeeze,
) -> pl.DataFrame:
    """Simulates stock price, earnings, reinvestment, and returns over a given period."""

    ts = history_ts(m, investors)
    history_months = 12 * m.history_length
    months = 12 * (m.years + m.history_length)

    reinvested = ts.monthly_earnings[history_months] * (1 - m.payout_ratio)

    for t in range(history_months + 1, months + 1):
        ts.monthly_earnings[t] = (
            ts.monthly_earnings[t - 1]
            * (1 + reinvested / ts.price_idx[t - 1])
            * (1 + ts.dz[t] * m.earnings_vol / SQRT_12)
        )
        reinvested = ts.monthly_earnings[t] * (1 - m.payout_ratio)
        ts.annualized_earnings[t] = (
            (1 + ts.monthly_earnings[t] / ts.price_idx[t - 1]) ** 12 - 1
        ) * ts.price_idx[t - 1]
        ts.total_cash[t] = (
            ts.total_cash[t - 1] + ts.monthly_earnings[t] * m.payout_ratio
        )
        ts.n_year_annualized_return[t] = weighted_avg_returns(
            ts.return_idx, return_weights(), t
        )
        ts.squeeze[t] = (
            squeeze_params.squeeze_target
            + squeeze_params.max_deviation
            * np.tanh(
                (ts.n_year_annualized_return[t] - m.initial_expected_return)
                / squeeze_params.squeezing
            )
        )
        ts.expected_return_x[t] = (
            ts.squeeze[t] * investors.speed_of_adjustment
            + (1 - investors.speed_of_adjustment) * ts.expected_return_x[t - 1]
        )
        ts.cash_post_distribution_x[t] = (
            ts.cash_x[t - 1]
            + (ts.monthly_earnings[t] * m.payout_ratio)
            * ts.equity_x[t - 1]
            / ts.price_idx[t - 1]
        )
        ts.cash_post_distribution_y[t] = (
            ts.cash_y[t - 1]
            + (ts.monthly_earnings[t] * m.payout_ratio)
            * ts.equity_y[t - 1]
            / ts.price_idx[t - 1]
        )
        ts.a[t] = (
            ts.expected_return_x[t]
            * ts.equity_x[t - 1]
            / (ts.price_idx[t - 1] * investors.gamma_x * investors.sigma_x**2)
            - 1
        )
        ts.b[t] = ts.annualized_earnings[t] * ts.equity_y[t - 1] / (
            ts.price_idx[t - 1] * investors.gamma_y * investors.sigma_y**2
        ) + ts.expected_return_x[t] * ts.cash_post_distribution_x[t] / (
            investors.gamma_x * investors.sigma_x**2
        )
        ts.c[t] = (
            ts.annualized_earnings[t]
            * ts.cash_post_distribution_y[t]
            / (investors.gamma_y * investors.sigma_y**2)
        )

        ts.price_idx[t] = quadratic(ts.a[t], ts.b[t], ts.c[t])
        ts.expected_return_y[t] = ts.annualized_earnings[t] / ts.price_idx[t]
        ts.return_idx[t] = ts.return_idx[t - 1] * (
            (ts.price_idx[t] + ts.monthly_earnings[t] * m.payout_ratio)
            / ts.price_idx[t - 1]
        )

        ts.wealth_x[t] = (
            ts.cash_x[t - 1]
            + ts.equity_x[t - 1] * ts.return_idx[t] / ts.return_idx[t - 1]
        )
        ts.wealth_y[t] = (
            ts.cash_y[t - 1]
            + ts.equity_y[t - 1] * ts.return_idx[t] / ts.return_idx[t - 1]
        )
        ts.equity_x[t] = (
            ts.wealth_x[t]
            * ts.expected_return_x[t]
            / (investors.gamma_x * investors.sigma_x**2)
        )
        ts.equity_y[t] = (
            ts.wealth_y[t]
            * ts.expected_return_y[t]
            / (investors.gamma_y * investors.sigma_y**2)
        )
        ts.cash_x[t] = ts.wealth_x[t] - ts.equity_x[t]
        ts.cash_y[t] = ts.wealth_y[t] - ts.equity_y[t]

    # Return results as a Polars DataFrame
    return pl.DataFrame(
        {
            "annualized_earnings": ts.annualized_earnings,
            "monthly_earnings": ts.monthly_earnings,
            "return_idx": ts.return_idx,
            "price_idx": ts.price_idx,
            "total_cash": ts.total_cash,
            "expected_return_y": ts.expected_return_y,
            "expected_return_x": ts.expected_return_x,
            "n_year_annualized_return": ts.n_year_annualized_return,
            "wealth_x": ts.wealth_x,
            "wealth_y": ts.wealth_y,
            "equity_x": ts.equity_x,
            "equity_y": ts.equity_y,
            "cash_x": ts.cash_x,
            "cash_y": ts.cash_y,
            "cash_post_distribution_x": ts.cash_post_distribution_x,
            "cash_post_distribution_y": ts.cash_post_distribution_y,
        }
    )
