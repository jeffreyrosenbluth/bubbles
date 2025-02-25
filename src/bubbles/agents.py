"""
An equilibrium multi-agent model in which the interaction between investors with different return
expectations produces a rich market dynamic that includes equity market volatility in excess of earnings volatility,
short-term trends, long-term mean reversion, high trading volume, and bubbles and crashes.
"""

import numpy as np
import polars as pl
from scipy.optimize import root_scalar

from bubbles.investors import weighted_avg_returns, weights_5_36
from bubbles.market import Market
from bubbles.protocols import InvestorProvider
from bubbles.timeseries import TimeSeries

SQRT_12 = np.sqrt(12)


def history_ts(
    mkt: Market,
    investors: list[InvestorProvider],
) -> TimeSeries:
    """Generate historical time series data for market initialization.

    This function simulates historical market data to establish initial conditions
    for the main simulation. It creates a baseline of price indices, earnings,
    and investor positions.

    Args:
        mkt: Market parameters
        investors: List of investor providers representing different trading strategies

    Returns:
        TimeSeries: Historical market data and investor positions
    """
    history_months = 12 * mkt.history_length
    ts = TimeSeries.initialize(mkt, investors)

    # Initialize `TimeSeries` for the `t = 1`
    ts.monthly_earnings[1] = (1 + mkt.initial_expected_return) ** (1 / 12) - 1
    reinvested = ts.monthly_earnings[1] * (1 - mkt.payout_ratio)
    ts.price_idx[1] = ts.price_idx[0] + reinvested
    ts.annualized_earnings[1] = ts.annualize(mkt, 1)
    ts.return_idx[1] = ts.calculate_return_idx(mkt, 1)

    # Initialize `TimeSeries` for the `t = 2` to `t = history_months`
    for t in range(2, history_months + 1):
        ts.monthly_earnings[t] = ts.earnings(mkt, reinvested, t)
        reinvested = ts.monthly_earnings[t] * (1 - mkt.payout_ratio)
        ts.price_idx[t] = ts.price_idx[t - 1] + reinvested
        ts.annualized_earnings[t] = ts.annualize(mkt, t)
        ts.return_idx[t] = ts.calculate_return_idx(mkt, t)

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


def market_clearing_error(price: float, t: int, ts: TimeSeries, mkt: Market) -> float:
    """Calculate the excess demand at a given price.

    Args:
        price: Proposed price index
        t: Current time
        ts: Time series data containing market and investor state

    Returns:
        np.float64: Excess demand (positive means demand > supply)
    """
    total_demand = 0.0
    for investor in ts.investors:
        # Calculate desired equity position for each investor
        match investor.investor_type():
            case "extrapolator":
                desired_equity = investor.desired_equity(
                    t, ts.n_year_annualized_return[t], mkt, ts.price_idx[t - 1], price
                )
            case "long_term":
                desired_equity = investor.desired_equity(
                    t, ts.annualized_earnings[t], ts.price_idx[t - 1], price
                )
            case "rebalancer_60_40":
                desired_equity = investor.desired_equity()
            case _:  # Add this wildcard case
                raise ValueError(f"Unknown investor type: {investor.investor_type()}")
        total_demand += desired_equity

    return total_demand - price  # Supply is just the price


def find_equilibrium_price(t: int, ts: TimeSeries, mkt: Market) -> float:
    """Find the equilibrium price using numerical root finding.

    Args:
        t: Current time
        ts: Time series data containing market and investor state
        mkt: Market parameters

    Returns:
        np.float64: Equilibrium price index
    """
    initial_guess = ts.price_idx[t - 1]

    def objective(price: float) -> float:
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
            investor.wealth()[t] = (
                investor.cash()[t - 1]
                + investor.equity()[t - 1] * ts.return_idx[t] / ts.return_idx[t - 1]
            )
            match investor.investor_type():
                case "extrapolator":
                    investor.expected_return()[t] = investor.calculate_expected_return(
                        t, ts.n_year_annualized_return[t], mkt
                    )
                    investor.equity()[t] = (
                        investor.wealth()[t]
                        * investor.expected_return()[t]
                        / (investor.gamma() * investor.sigma() ** 2)
                    )
                case "long_term":
                    investor.expected_return()[t] = investor.calculate_expected_return(
                        ts.annualized_earnings[t],
                        ts.price_idx[t],
                    )
                    investor.equity()[t] = (
                        investor.wealth()[t]
                        * investor.expected_return()[t]
                        / (investor.gamma() * investor.sigma() ** 2)
                    )
                case "rebalancer_60_40":
                    investor.expected_return()[t] = investor.calculate_expected_return()
                    investor.equity()[t] = investor.wealth()[t] * investor.desired_equity()

            investor.cash()[t] = investor.wealth()[t] - investor.equity()[t]

        fair_value = ts.annualized_earnings / mkt.initial_expected_return

    # Return results as a Polars DataFrame
    return pl.DataFrame(
        {
            "Month": list(range(len(ts.annualized_earnings))),
            # "Annualized E": ts.annualized_earnings,
            # "Monthly E": ts.monthly_earnings,
            "Return Idx": ts.return_idx,
            "Price Idx": ts.price_idx,
            "Premium": np.log(ts.price_idx / fair_value),
            "Wealth x": ts.investors[0].wealth(),
            "Wealth y": ts.investors[1].wealth(),
            "Wealth z": ts.investors[2].wealth(),
            "Equity x": ts.investors[0].equity(),
            "Equity y": ts.investors[1].equity(),
            "Equity z": ts.investors[2].equity(),
            "Cash x": ts.investors[0].cash(),
            "Cash y": ts.investors[1].cash(),
            "Cash z": ts.investors[2].cash(),
            "kappa_x": ts.investors[0].equity() / ts.investors[0].wealth(),
            "kappa_y": ts.investors[1].equity() / ts.investors[1].wealth(),
            "kappa_z": ts.investors[2].equity() / ts.investors[2].wealth(),
            "Fair Value": fair_value,
        }
    )
