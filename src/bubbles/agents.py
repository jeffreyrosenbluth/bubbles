import numpy as np
import polars as pl

from bubbles.core import (
    InvestorParameters,
    MarketParameters,
    SqueezeParameters,
    TimeSeries,
    initialize_time_series,
    return_weights,
    weighted_avg_returns,
)

SQRT_12 = np.sqrt(12)


def quadratic(a: float, b: float, c: float) -> float:
    discriminant = b**2 - 4 * a * c
    return (-b - np.sqrt(discriminant)) / (2 * a)


def monthly_earnings_next(
    monthly_earnings_prev: float,
    reinvested: float,
    price_idx_pred: float,
    dz: float,
    earnings_vol,
) -> float:
    return (
        monthly_earnings_prev
        * (1 + reinvested / price_idx_pred)
        * (1 + dz * earnings_vol / SQRT_12)
    )


def annualize(monthly_earnings: float, price_idx_prev: float) -> float:
    return ((1 + monthly_earnings / price_idx_prev) ** 12 - 1) * price_idx_prev


def normalize(
    n_year_annualized_return: float,
    initial_expected_return: float,
    squeeze_params: SqueezeParameters,
) -> float:
    return squeeze_params.squeeze_target + squeeze_params.max_deviation * np.tanh(
        (n_year_annualized_return - initial_expected_return) / squeeze_params.squeezing
    )


def return_idx_next(
    return_idx_prev: float,
    monthly_earnings: float,
    price_idx_prev: float,
    price_idx: float,
    payout_ratio: float,
) -> float:
    return return_idx_prev * (
        (price_idx + monthly_earnings * payout_ratio) / price_idx_prev
    )


def history_ts(
    m: MarketParameters,
    investors: InvestorParameters,
) -> TimeSeries:
    history_months = 12 * m.history_length
    s = initialize_time_series(m)
    s.monthly_earnings[1] = (1 + m.initial_expected_return) ** (1 / 12) - 1
    reinvested = s.monthly_earnings[1] * (1 - m.payout_ratio)
    s.price_idx[1] = s.price_idx[0] + reinvested
    s.annualized_earnings[1] = annualize(s.monthly_earnings[1], s.price_idx[0])
    s.return_idx[1] = return_idx_next(
        s.return_idx[0],
        s.monthly_earnings[1],
        s.price_idx[0],
        s.price_idx[1],
        m.payout_ratio,
    )
    s.dz[: history_months + 1] = 0

    for t in range(2, history_months + 1):
        s.monthly_earnings[t] = monthly_earnings_next(
            s.monthly_earnings[t - 1],
            reinvested,
            s.price_idx[t - 1],
            s.dz[t],
            m.earnings_vol,
        )
        reinvested = s.monthly_earnings[t] * (1 - m.payout_ratio)

        s.price_idx[t] = s.price_idx[t - 1] + reinvested
        s.annualized_earnings[t] = annualize(s.monthly_earnings[t], s.price_idx[t - 1])
        s.return_idx[t] = return_idx_next(
            s.return_idx[t - 1],
            s.monthly_earnings[t],
            s.price_idx[t - 1],
            s.price_idx[t],
            m.payout_ratio,
        )

    s.expected_return_y[history_months] = (
        s.annualized_earnings[history_months] / s.price_idx[history_months]
    )
    s.n_year_annualized_return[history_months] = s.expected_return_y[history_months]

    merton_share_y = (
        s.annualized_earnings[history_months]
        / s.price_idx[history_months]
        / (investors.gamma_y * investors.sigma_y**2)
    )
    merton_share_x = (
        s.annualized_earnings[history_months]
        / s.price_idx[history_months]
        / (investors.gamma_x * investors.sigma_x**2)
    )

    starting_wealth = s.price_idx[history_months] / (
        investors.percent_y * merton_share_y + investors.percent_x * merton_share_x
    )

    s.total_cash[history_months] = starting_wealth - s.price_idx[history_months]
    s.expected_return_x[history_months] = s.expected_return_y[history_months]
    s.wealth_x[history_months] = investors.percent_x * starting_wealth
    s.wealth_y[history_months] = investors.percent_y * starting_wealth
    s.equity_x[history_months] = (
        s.wealth_x[history_months]
        * s.expected_return_x[history_months]
        / (investors.gamma_x * investors.sigma_x**2)
    )
    s.equity_y[history_months] = (
        s.wealth_y[history_months]
        * s.expected_return_y[history_months]
        / (investors.gamma_y * investors.sigma_y**2)
    )
    s.cash_x[history_months] = s.wealth_x[history_months] - s.equity_x[history_months]
    s.cash_y[history_months] = s.wealth_y[history_months] - s.equity_y[history_months]
    return s


def data_table(
    m: MarketParameters,
    investors: InvestorParameters,
    squeeze_params: SqueezeParameters,
) -> pl.DataFrame:
    """Simulates stock price, earnings, reinvestment, and returns over a given period."""

    ts = history_ts(m, investors)
    history_months = 12 * m.history_length
    months = 12 * (m.years + m.history_length)

    reinvested = ts.monthly_earnings[history_months] * (1 - m.payout_ratio)

    for t in range(history_months + 1, months + 1):
        ts.monthly_earnings[t] = monthly_earnings_next(
            ts.monthly_earnings[t - 1],
            reinvested,
            ts.price_idx[t - 1],
            ts.dz[t],
            m.earnings_vol,
        )
        reinvested = ts.monthly_earnings[t] * (1 - m.payout_ratio)
        ts.annualized_earnings[t] = annualize(
            ts.monthly_earnings[t], ts.price_idx[t - 1]
        )
        ts.total_cash[t] = (
            ts.total_cash[t - 1] + ts.monthly_earnings[t] * m.payout_ratio
        )
        ts.n_year_annualized_return[t] = weighted_avg_returns(
            ts.return_idx, return_weights(), t
        )
        ts.squeeze[t] = normalize(
            ts.n_year_annualized_return[t], m.initial_expected_return, squeeze_params
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
