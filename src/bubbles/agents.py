"""
An equilibrium 2-agent model in which the interaction between investors with different return
expectations produces a rich market dynamic that includes equity market volatility in excess of earnings volatility,
short-term trends, long-term mean reversion, high trading volume, and bubbles and crashes.
"""

import numpy as np
import polars as pl

from bubbles.core import InvestorProvider, Market, TimeSeries, weighted_avg_returns, weights_5_36

SQRT_12 = np.sqrt(12)


def quadratic(a: float, b: float, c: float) -> float:
    """Solve quadratic equation of the form ax² + bx + c = 0, returning the positive root.

    Args:
        a: Coefficient of x²
        b: Coefficient of x
        c: Constant term

    Returns:
        float: The positive root of the quadratic equation

    Raises:
        ValueError: If the discriminant is negative (no real solutions)
    """
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("Quadratic equation has no real solutions")
    return (-b - np.sqrt(discriminant)) / (2 * a)


def history_ts(
    mp: Market,
    investors: list[InvestorProvider],
) -> TimeSeries:
    """Generate historical time series data for market initialization.

    This function simulates historical market data to establish initial conditions
    for the main simulation. It creates a baseline of price indices, earnings,
    and investor positions.

    Args:
        mp: Market parameters
        investors: List of investor providers representing different trading strategies

    Returns:
        TimeSeries: Historical market data and investor positions
    """
    history_months = 12 * mp.history_length
    ts = TimeSeries.initialize(mp, investors)

    ts.monthly_earnings[1] = (1 + mp.initial_expected_return) ** (1 / 12) - 1
    reinvested = ts.monthly_earnings[1] * (1 - mp.payout_ratio)
    ts.price_idx[1] = ts.price_idx[0] + reinvested
    ts.annualized_earnings[1] = ts.annualize(mp, 1)
    ts.return_idx[1] = ts.calculate_return_idx(mp, 1)

    for t in range(2, history_months + 1):
        ts.monthly_earnings[t] = ts.earnings(mp, reinvested, t)
        reinvested = ts.monthly_earnings[t] * (1 - mp.payout_ratio)
        ts.price_idx[t] = ts.price_idx[t - 1] + reinvested
        ts.annualized_earnings[t] = ts.annualize(mp, t)
        ts.return_idx[t] = ts.calculate_return_idx(mp, t)

    for inv in ts.investors:
        inv.get_expected_return()[history_months] = (
            ts.annualized_earnings[history_months] / ts.price_idx[history_months]
        )

    ts.n_year_annualized_return[history_months] = (
        ts.annualized_earnings[history_months] / ts.price_idx[history_months]
    )

    starting_wealth = ts.price_idx[history_months] / sum(
        inv.get_percent()
        * inv.merton_share(ts.annualized_earnings[history_months] / ts.price_idx[history_months])
        for inv in ts.investors
    )

    ts.total_cash[history_months] = starting_wealth - ts.price_idx[history_months]

    for inv in ts.investors:
        inv.get_wealth()[history_months] = inv.get_percent() * starting_wealth
        inv.get_equity()[history_months] = (
            inv.get_wealth()[history_months]
            * inv.get_expected_return()[history_months]
            / (inv.get_gamma() * inv.get_sigma() ** 2)
        )
        inv.get_cash()[history_months] = (
            inv.get_wealth()[history_months] - inv.get_equity()[history_months]
        )
    return ts


def calculate_quadratic_coefficients(
    t: int,
    ts: TimeSeries,
) -> tuple[float, float, float]:
    """Calculate coefficients for the quadratic equation determining the next price index.

    The quadratic equation balances supply and demand in the market between different
    investor types to determine the equilibrium price.

    Args:
        t: Current time step
        ts: Time series data containing market and investor state

    Returns:
        tuple[float, float, float]: Coefficients (a, b, c) for the quadratic equation ax² + bx + c = 0
    """
    a = (
        ts.investors[0].get_expected_return()[t]
        * ts.investors[0].get_equity()[t - 1]
        / (ts.price_idx[t - 1] * ts.investors[0].get_gamma() * ts.investors[0].get_sigma() ** 2)
        - 1
    )

    b = ts.annualized_earnings[t] * ts.investors[1].get_equity()[t - 1] / (
        ts.price_idx[t - 1] * ts.investors[1].get_gamma() * ts.investors[1].get_sigma() ** 2
    ) + ts.investors[0].get_expected_return()[t] * ts.investors[0].get_cash_post_distribution()[
        t
    ] / (ts.investors[0].get_gamma() * ts.investors[0].get_sigma() ** 2)

    c = (
        ts.annualized_earnings[t]
        * ts.investors[1].get_cash_post_distribution()[t]
        / (ts.investors[1].get_gamma() * ts.investors[1].get_sigma() ** 2)
    )

    return a, b, c


def data_table(
    m: Market,
    investors: list[InvestorProvider],
) -> pl.DataFrame:
    """Simulate market dynamics and return results as a DataFrame.

    This function performs the main market simulation, modeling the interaction
    between different types of investors and their impact on market prices and returns.

    Args:
        m: Market parameters including simulation length and initial conditions
        investors: List of investor providers representing different trading strategies

    Returns:
        pl.DataFrame: DataFrame containing simulation results including:
            - Month: Time index (0 to simulation length)
            - Annualized E: Annualized earnings
            - Monthly E: Monthly earnings
            - Return Idx: Return index tracking total returns
            - Price Idx: Price index tracking market prices
            - Premium: Log premium over fair value
            - Expected Return x/y: Expected returns for each investor type
            - Wealth x/y: Total wealth for each investor type
            - Equity x/y: Equity holdings for each investor type
            - Cash x/y: Cash holdings for each investor type
            - Kappa x/y: Portfolio allocation ratios for each investor type
            - Fair Value: Theoretical fair value based on earnings
    """

    ts = history_ts(m, investors)
    history_months = 12 * m.history_length
    months = 12 * (m.years + m.history_length)
    np.random.seed(m.seed)

    reinvested = ts.monthly_earnings[history_months] * (1 - m.payout_ratio)

    for t in range(history_months + 1, months + 1):
        ts.monthly_earnings[t] = ts.earnings(m, reinvested, t)
        reinvested = ts.monthly_earnings[t] * (1 - m.payout_ratio)
        ts.annualized_earnings[t] = ts.annualize(m, t)
        ts.total_cash[t] = ts.total_cash[t - 1] + ts.monthly_earnings[t] * m.payout_ratio
        ts.n_year_annualized_return[t] = weighted_avg_returns(ts.return_idx, weights_5_36(), t)
        ts.squeeze[t] = ts.investors[0].normalize_weights(
            ts.n_year_annualized_return[t], m.initial_expected_return
        )
        ts.investors[0].get_expected_return()[t] = (
            ts.squeeze[t] * ts.investors[0].speed_of_adjustment
            + (1 - ts.investors[0].speed_of_adjustment)
            * ts.investors[0].get_expected_return()[t - 1]
        )

        for investor in investors:
            investor.get_cash_post_distribution()[t] = (
                investor.get_cash()[t - 1]
                + (ts.monthly_earnings[t] * m.payout_ratio)
                * investor.get_equity()[t - 1]
                / ts.price_idx[t - 1]
            )

        ts.a[t], ts.b[t], ts.c[t] = calculate_quadratic_coefficients(t, ts)

        ts.price_idx[t] = quadratic(ts.a[t], ts.b[t], ts.c[t])
        ts.investors[1].get_expected_return()[t] = ts.annualized_earnings[t] / ts.price_idx[t]
        ts.return_idx[t] = ts.return_idx[t - 1] * (
            (ts.price_idx[t] + ts.monthly_earnings[t] * m.payout_ratio) / ts.price_idx[t - 1]
        )
        for investor in investors:
            investor.get_wealth()[t] = (
                investor.get_cash()[t - 1]
                + investor.get_equity()[t - 1] * ts.return_idx[t] / ts.return_idx[t - 1]
            )
            investor.get_equity()[t] = (
                investor.get_wealth()[t]
                * investor.get_expected_return()[t]
                / (investor.get_gamma() * investor.get_sigma() ** 2)
            )
            investor.get_cash()[t] = investor.get_wealth()[t] - investor.get_equity()[t]

        fair_value = ts.annualized_earnings / m.initial_expected_return

    # Return results as a Polars DataFrame
    return pl.DataFrame(
        {
            "Month": list(range(len(ts.annualized_earnings))),
            "Annualized E": ts.annualized_earnings,
            "Monthly E": ts.monthly_earnings,
            "Return Idx": ts.return_idx,
            "Price Idx": ts.price_idx,
            "Premium": np.log(ts.price_idx / fair_value),
            "Expected Return y": ts.investors[1].get_expected_return(),
            "Expected Return x": ts.investors[0].get_expected_return(),
            "Wealth x": ts.investors[0].get_wealth(),
            "Wealth y": ts.investors[1].get_wealth(),
            "Equity x": ts.investors[0].get_equity(),
            "Equity y": ts.investors[1].get_equity(),
            "Cash x": ts.investors[0].get_cash(),
            "Cash y": ts.investors[1].get_cash(),
            "Kappa_x": ts.investors[0].get_equity() / ts.investors[0].get_wealth(),
            "Kappa_y": ts.investors[1].get_equity() / ts.investors[1].get_wealth(),
            "Fair Value": fair_value,
        }
    )
