"""
An equilibrium 2-agent model in which the interaction between investors with different return
expectations produces a rich market dynamic that includes equity market volatility in excess of earnings volatility,
short-term trends, long-term mean reversion, high trading volume, and bubbles and crashes.
"""

import numpy as np
import polars as pl
from scipy.optimize import root_scalar

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

    # Initialize `TimeSeries` for the `t = 1`
    ts.monthly_earnings[1] = (1 + mp.initial_expected_return) ** (1 / 12) - 1
    reinvested = ts.monthly_earnings[1] * (1 - mp.payout_ratio)
    ts.price_idx[1] = ts.price_idx[0] + reinvested
    ts.annualized_earnings[1] = ts.annualize(mp, 1)
    ts.return_idx[1] = ts.calculate_return_idx(mp, 1)

    # Initialize `TimeSeries` for the `t = 2` to `t = history_months`
    for t in range(2, history_months + 1):
        ts.monthly_earnings[t] = ts.earnings(mp, reinvested, t)
        reinvested = ts.monthly_earnings[t] * (1 - mp.payout_ratio)
        ts.price_idx[t] = ts.price_idx[t - 1] + reinvested
        ts.annualized_earnings[t] = ts.annualize(mp, t)
        ts.return_idx[t] = ts.calculate_return_idx(mp, t)

    ts.n_year_annualized_return[history_months] = (
        ts.annualized_earnings[history_months] / ts.price_idx[history_months]
    )

    current_return = ts.annualized_earnings[history_months] / ts.price_idx[history_months]
    total_percent_equity = sum(
        inv.percent() * inv.merton_share(current_return) for inv in ts.investors
    )

    # Calculate starting_wealth after we have the complete sum
    starting_wealth = ts.price_idx[history_months] / total_percent_equity

    for inv in ts.investors:
        # Calculate investor positions
        inv.wealth()[history_months] = inv.percent() * starting_wealth
        inv.equity()[history_months] = (
            inv.wealth()[history_months]
            * ts.annualized_earnings[history_months]
            / ts.price_idx[history_months]
            / (inv.gamma() * inv.sigma() ** 2)
        )
        inv.cash()[history_months] = inv.wealth()[history_months] - inv.equity()[history_months]
        inv.expected_return()[history_months] = (
            ts.annualized_earnings[history_months] / ts.price_idx[history_months]
        )

    ts.total_cash[history_months] = starting_wealth - ts.price_idx[history_months]
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
        ts.investors[0].expected_return()[t]
        * ts.investors[0].equity()[t - 1]
        / (ts.price_idx[t - 1] * ts.investors[0].gamma() * ts.investors[0].sigma() ** 2)
        - 1
    )

    b = ts.annualized_earnings[t] * ts.investors[1].equity()[t - 1] / (
        ts.price_idx[t - 1] * ts.investors[1].gamma() * ts.investors[1].sigma() ** 2
    ) + ts.investors[0].expected_return()[t] * ts.investors[0].cash_post_distribution()[t] / (
        ts.investors[0].gamma() * ts.investors[0].sigma() ** 2
    )

    c = (
        ts.annualized_earnings[t]
        * ts.investors[1].cash_post_distribution()[t]
        / (ts.investors[1].gamma() * ts.investors[1].sigma() ** 2)
    )

    return a, b, c


def market_clearing_error(price: float, t: int, ts: TimeSeries, mkt: Market) -> float:
    """Calculate the excess demand at a given price.

    Args:
        price: Proposed price index
        t: Current time step
        ts: Time series data containing market and investor state

    Returns:
        float: Excess demand (positive means demand > supply)
    """
    total_demand = 0
    for investor in ts.investors:
        # Calculate desired equity position for each investor
        expected_return = investor.calculate_expected_return(t, ts, mkt, price)
        cash = investor.cash_post_distribution()[t]
        desired_equity = (
            expected_return * (cash + price * investor.equity()[t - 1] / ts.price_idx[t - 1])
        ) / (investor.gamma() * investor.sigma() ** 2)
        total_demand += desired_equity

    return total_demand - price  # Supply is just the price


def find_equilibrium_price(t: int, ts: TimeSeries, mkt: Market) -> float:
    """Find the equilibrium price using numerical root finding.

    Args:
        t: Current time
        ts: Time series data containing market and investor state

    Returns:
        float: Equilibrium price index
    """
    initial_guess = ts.price_idx[t - 1]

    def objective(price):
        return market_clearing_error(price, t, ts, mkt)

    result = root_scalar(
        objective,
        x0=initial_guess,
        x1=initial_guess * 1.1,
        method="secant",
        xtol=1e-6,
    )

    if not result.converged:
        raise ValueError("Failed to find equilibrium price")

    return result.root


def data_table(
    mkt: Market,
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

    ts = history_ts(mkt, investors)
    history_months = 12 * mkt.history_length
    months = 12 * (mkt.years + mkt.history_length)
    np.random.seed(mkt.seed)

    reinvested = ts.monthly_earnings[history_months] * (1 - mkt.payout_ratio)

    for t in range(history_months + 1, months + 1):
        ts.monthly_earnings[t] = ts.earnings(mkt, reinvested, t)
        reinvested = ts.monthly_earnings[t] * (1 - mkt.payout_ratio)
        ts.annualized_earnings[t] = ts.annualize(mkt, t)
        ts.total_cash[t] = ts.total_cash[t - 1] + ts.monthly_earnings[t] * mkt.payout_ratio
        ts.n_year_annualized_return[t] = weighted_avg_returns(ts.return_idx, weights_5_36(), t)

        for investor in investors:
            investor.cash_post_distribution()[t] = (
                investor.cash()[t - 1]
                + (ts.monthly_earnings[t] * mkt.payout_ratio)
                * investor.equity()[t - 1]
                / ts.price_idx[t - 1]
            )

        ts.price_idx[t] = find_equilibrium_price(t, ts, mkt)

        ts.return_idx[t] = ts.return_idx[t - 1] * (
            (ts.price_idx[t] + ts.monthly_earnings[t] * mkt.payout_ratio) / ts.price_idx[t - 1]
        )

        for investor in investors:
            investor.expected_return()[t] = investor.calculate_expected_return(
                t, ts, mkt, ts.price_idx[t]
            )
            investor.wealth()[t] = (
                investor.cash()[t - 1]
                + investor.equity()[t - 1] * ts.return_idx[t] / ts.return_idx[t - 1]
            )
            investor.equity()[t] = (
                investor.wealth()[t]
                * investor.expected_return()[t]
                / (investor.gamma() * investor.sigma() ** 2)
            )
            investor.cash()[t] = investor.wealth()[t] - investor.equity()[t]

        fair_value = ts.annualized_earnings / mkt.initial_expected_return

    # Return results as a Polars DataFrame
    return pl.DataFrame(
        {
            "Month": list(range(len(ts.annualized_earnings))),
            "Annualized E": ts.annualized_earnings,
            "Monthly E": ts.monthly_earnings,
            "Return Idx": ts.return_idx,
            "Price Idx": ts.price_idx,
            "Premium": np.log(ts.price_idx / fair_value),
            "Expected Return y": ts.investors[1].expected_return(),
            "Expected Return x": ts.investors[0].expected_return(),
            "Wealth x": ts.investors[0].wealth(),
            "Wealth y": ts.investors[1].wealth(),
            "Equity x": ts.investors[0].equity(),
            "Equity y": ts.investors[1].equity(),
            "Cash x": ts.investors[0].cash(),
            "Cash y": ts.investors[1].cash(),
            "Kappa_x": ts.investors[0].equity() / ts.investors[0].wealth(),
            "Kappa_y": ts.investors[1].equity() / ts.investors[1].wealth(),
            "Fair Value": fair_value,
        }
    )
