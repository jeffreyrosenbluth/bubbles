from dataclasses import dataclass, field

import numpy as np
import polars as pl

from bubbles.dz import dz

SQRT_12 = np.sqrt(12)


@dataclass
class Model:
    shares_ex: float
    shares_lt: float
    cash_ex: float
    cash_lt: float
    weight_ex: float
    gamma_ex: float
    gamma_lt: float
    sigma_ex: float
    sigma_lt: float
    earnings: float
    alpha: float
    squeeze_target: float
    max_deviation: float
    expected_return_0: float
    squeezing: float


@dataclass
class Market:
    initial_expected_return: float = 0.04
    earnings_vol: float = 0.10
    payout_ratio: float = 0.5


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


def data_table(
    months: int,
    r0: float,
    payout_ratio: float,
    earnings_vol: float,
    history_length: int,
    investors: Investors,
    squeeze_params: Squeeze,
) -> pl.DataFrame:
    """Simulates stock price, earnings, reinvestment, and returns over a given period."""

    SQRT_12 = np.sqrt(12)
    history_months = 12 * history_length

    monthly_earnings = np.full(months + 1, np.nan, dtype=float)
    price_idx = np.ones(months + 1)
    annualized_earnings = np.full(months + 1, np.nan, dtype=float)
    return_idx = np.ones(months + 1)
    total_cash = np.full(months + 1, np.nan, dtype=float)
    expected_return_x = np.full(months + 1, np.nan, dtype=float)
    expected_return_y = np.full(months + 1, np.nan, dtype=float)
    squeeze = np.full(months + 1, np.nan, dtype=float)
    wealth_x = np.full(months + 1, np.nan, dtype=float)
    wealth_y = np.full(months + 1, np.nan, dtype=float)
    equity_x = np.full(months + 1, np.nan, dtype=float)
    equity_y = np.full(months + 1, np.nan, dtype=float)
    cash_x = np.full(months + 1, np.nan, dtype=float)
    cash_y = np.full(months + 1, np.nan, dtype=float)
    cash_post_distribution_x = np.full(months + 1, np.nan, dtype=float)
    cash_post_distribution_y = np.full(months + 1, np.nan, dtype=float)
    n_year_annualized_return = np.full(months + 1, np.nan, dtype=float)
    a = np.full(months + 1, np.nan, dtype=float)
    b = np.full(months + 1, np.nan, dtype=float)
    c = np.full(months + 1, np.nan, dtype=float)
    z = dz.copy()

    # Initial conditions
    monthly_earnings[1] = (1 + r0) ** (1 / 12) - 1
    reinvested = monthly_earnings[1] * (1 - payout_ratio)
    price_idx[1] = price_idx[0] + reinvested
    annualized_earnings[1] = (
        (1 + monthly_earnings[1] / price_idx[0]) ** 12 - 1
    ) * price_idx[0]
    return_idx[1] = return_idx[0] * (
        (price_idx[1] + monthly_earnings[1] - reinvested) / price_idx[0]
    )
    z[: history_months + 1] = 0  # Zero-out initial noise

    for t in range(2, history_months + 1):
        monthly_earnings[t] = (
            monthly_earnings[t - 1]
            * (1 + reinvested / price_idx[t - 1])
            * (1 + z[t] * earnings_vol / SQRT_12)
        )
        reinvested = monthly_earnings[t] * (1 - payout_ratio)

        price_idx[t] = price_idx[t - 1] + reinvested
        annualized_earnings[t] = (
            (1 + monthly_earnings[t] / price_idx[t - 1]) ** 12 - 1
        ) * price_idx[t - 1]
        return_idx[t] = return_idx[t - 1] * (
            (price_idx[t] + monthly_earnings[t] - reinvested) / price_idx[t - 1]
        )

    expected_return_y[history_months] = (
        annualized_earnings[history_months] / price_idx[history_months]
    )
    n_year_annualized_return[history_months] = expected_return_y[history_months]

    # Merton shares
    merton_share_y = (
        annualized_earnings[history_months]
        / price_idx[history_months]
        / (investors.gamma_y * investors.sigma_y**2)
    )
    merton_share_x = (
        annualized_earnings[history_months]
        / price_idx[history_months]
        / (investors.gamma_x * investors.sigma_x**2)
    )

    starting_wealth = price_idx[history_months] / (
        investors.percent_y * merton_share_y + investors.percent_x * merton_share_x
    )

    total_cash[history_months] = starting_wealth - price_idx[history_months]
    expected_return_x[history_months] = expected_return_y[history_months]
    wealth_x[history_months] = investors.percent_x * starting_wealth
    wealth_y[history_months] = investors.percent_y * starting_wealth
    equity_x[history_months] = (
        wealth_x[history_months]
        * expected_return_x[history_months]
        / (investors.gamma_x * investors.sigma_x**2)
    )
    equity_y[history_months] = (
        wealth_y[history_months]
        * expected_return_y[history_months]
        / (investors.gamma_y * investors.sigma_y**2)
    )
    cash_x[history_months] = wealth_x[history_months] - equity_x[history_months]
    cash_y[history_months] = wealth_y[history_months] - equity_y[history_months]

    for t in range(history_months + 1, months + 1):
        monthly_earnings[t] = (
            monthly_earnings[t - 1]
            * (1 + reinvested / price_idx[t - 1])
            * (1 + z[t] * earnings_vol / SQRT_12)
        )
        reinvested = monthly_earnings[t] * (1 - payout_ratio)
        annualized_earnings[t] = (
            (1 + monthly_earnings[t] / price_idx[t - 1]) ** 12 - 1
        ) * price_idx[t - 1]
        total_cash[t] = total_cash[t - 1] + monthly_earnings[t] * payout_ratio
        n_year_annualized_return[t] = weighted_avg_returns(
            return_idx, return_weights(), t
        )
        squeeze[t] = (
            squeeze_params.squeeze_target
            + squeeze_params.max_deviation
            * np.tanh((n_year_annualized_return[t] - r0) / squeeze_params.squeezing)
        )
        expected_return_x[t] = (
            squeeze[t] * investors.speed_of_adjustment
            + (1 - investors.speed_of_adjustment) * expected_return_x[t - 1]
        )
        cash_post_distribution_x[t] = (
            cash_x[t - 1]
            + (monthly_earnings[t] * payout_ratio) * equity_x[t - 1] / price_idx[t - 1]
        )
        cash_post_distribution_y[t] = (
            cash_y[t - 1]
            + (monthly_earnings[t] * payout_ratio) * equity_y[t - 1] / price_idx[t - 1]
        )
        a[t] = (
            expected_return_x[t]
            * equity_x[t - 1]
            / (price_idx[t - 1] * investors.gamma_x * investors.sigma_x**2)
            - 1
        )
        b[t] = annualized_earnings[t] * equity_y[t - 1] / (
            price_idx[t - 1] * investors.gamma_y * investors.sigma_y**2
        ) + expected_return_x[t] * cash_post_distribution_x[t] / (
            investors.gamma_x * investors.sigma_x**2
        )
        c[t] = (
            annualized_earnings[t]
            * cash_post_distribution_y[t]
            / (investors.gamma_y * investors.sigma_y**2)
        )

        price_idx[t] = quadratic(a[t], b[t], c[t])
        expected_return_y[t] = annualized_earnings[t] / price_idx[t]
        return_idx[t] = return_idx[t - 1] * (
            (price_idx[t] + monthly_earnings[t] * payout_ratio) / price_idx[t - 1]
        )

        wealth_x[t] = (
            cash_x[t - 1] + equity_x[t - 1] * return_idx[t] / return_idx[t - 1]
        )
        wealth_y[t] = (
            cash_y[t - 1] + equity_y[t - 1] * return_idx[t] / return_idx[t - 1]
        )
        equity_x[t] = (
            wealth_x[t]
            * expected_return_x[t]
            / (investors.gamma_x * investors.sigma_x**2)
        )
        equity_y[t] = (
            wealth_y[t]
            * expected_return_y[t]
            / (investors.gamma_y * investors.sigma_y**2)
        )
        cash_x[t] = wealth_x[t] - equity_x[t]
        cash_y[t] = wealth_y[t] - equity_y[t]
    # Return results as a Polars DataFrame
    return pl.DataFrame(
        {
            "monthly_earnings": monthly_earnings,
            "price_idx": price_idx,
            "annualized_earnings": annualized_earnings,
            "return_idx": return_idx,
            "total_cash": total_cash,
            "n_year_annualized_return": n_year_annualized_return,
            "squeeze": squeeze,
            "expected_return_y": expected_return_y,
            "expected_return_x": expected_return_x,
            "wealth_x": wealth_x,
            "wealth_y": wealth_y,
            "equity_x": equity_x,
            "equity_y": equity_y,
            "cash_x": cash_x,
            "cash_y": cash_y,
            "cash_post_distribution_x": cash_post_distribution_x,
            "cash_post_distribution_y": cash_post_distribution_y,
            "a": a,
            "b": b,
            "c": c,
        }
    )


def returns_ex(m: Model, annual_returns: np.ndarray, psi: list) -> float:
    s = 0
    n = len(psi)
    for i in range(len(psi)):
        s += psi[i] * annual_returns[n - i]
    return m.squeeze_target + m.max_deviationm * np.tanh(
        (s - m.expected_return_0) / m.squeezing
    )


def weight_ex(m: Model, returns: np.ndarray) -> float:
    t = len(returns)
    s = 0
    for i in range(t):
        s += (1 - m.alpha) ** (i + 1) * m.alpha * returns[i]
    return s + m.weight_ex * (1 - m.alpha) ** t


def price(m: Model) -> float:
    weight_ex = (1 - m.alpha) * m.weight_ex + m.alpha * m.earnings / (
        m.gamma_ex * m.sigma_ex**2
    )
    a = m.shares_ex * weight_ex - 1
    b = m.shares_lt * m.earnings / (m.gamma_lt * m.sigma_lt**2) + m.cash_ex * weight_ex
    c = m.cash_lt * m.earnings / (m.gamma_lt * m.sigma_lt**2)

    discriminant = b**2 - 4 * a * c
    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
    root2 = (-b - np.sqrt(discriminant)) / (2 * a)
    return np.where(root1 > 0, root1, root2)


def cash(cash: float, earnings: float, shares: float, payout_rate: float) -> float:
    return cash + earnings / 12 + shares * payout_rate


def wealth(price: float, shares: float, cash: float) -> float:
    return cash + price * shares
