"""
An equilibrium multi-agent model.
"""

import numpy as np
import polars as pl
from scipy.optimize import root_scalar

from bubbles.core import InvestorProvider, Market
from bubbles.investors import Extrapolator
from bubbles.timeseries import TimeSeries

SQRT_12 = np.sqrt(12)


def history_ts(
    mkt: Market,
    investors: list[InvestorProvider],
) -> TimeSeries:
    """Generate historical time series data for market initialization.

    This function simulates historical market data to establish initial conditions
    for the main simulation.

    Args:
        mkt: Market parameters
        investors: List of investor providers representing different trading strategies

    Returns:
        TimeSeries: Historical market data and investor positions
    """
    total_percent = sum(inv.percent() for inv in investors)
    assert np.isclose(total_percent, 1.0), f"Total percent must be 1.0, got {total_percent}"

    history_months = 12 * mkt.history_length
    ts = TimeSeries.initialize(mkt, investors)

    # Initialize `TimeSeries` for `t = 1`
    ts.monthly_earnings[1] = (1 + mkt.initial_expected_return) ** (1 / 12) - 1
    reinvested = ts.monthly_earnings[1] * (1 - mkt.payout_ratio)
    ts.price_idx[1] = ts.price_idx[0] + reinvested
    ts.return_idx[1] = ts.calculate_return_idx(mkt, 1)

    # Initialize `TimeSeries` for `t = 2` to `t = history_months`
    for t in range(2, history_months + 1):
        ts.monthly_earnings[t] = ts.earnings(mkt, reinvested, t)
        reinvested = ts.monthly_earnings[t] * (1 - mkt.payout_ratio)
        ts.price_idx[t] = ts.price_idx[t - 1] + reinvested
        ts.return_idx[t] = ts.calculate_return_idx(mkt, t)

    annualized_earnings = ts.annualize(mkt, history_months)

    current_return = annualized_earnings / ts.price_idx[history_months]
    total_percent_equity = sum(
        inv.percent() * inv.merton_share(current_return) for inv in ts.investors
    )

    # Calculate starting_wealth after we have the complete sum
    starting_wealth = ts.price_idx[history_months] / total_percent_equity

    for investor in ts.investors:
        # Calculate investor positions
        if investor.investor_type() == "extrapolator":
            investor.expected_return = annualized_earnings / ts.price_idx[history_months]
        investor.wealth()[history_months] = investor.percent() * starting_wealth
        investor.equity()[history_months] = (
            investor.wealth()[history_months]
            * annualized_earnings
            / ts.price_idx[history_months]
            / (investor.gamma() * investor.sigma() ** 2)
        )
        investor.cash()[history_months] = (
            investor.wealth()[history_months] - investor.equity()[history_months]
        )
    return ts


def market_clearing_error(price: float, t: int, ts: TimeSeries, mkt: Market, noise: float) -> float:
    """Calculate the excess demand at a given price.

    Args:
        price: Proposed price index
        t: Current time
        ts: Time series data containing market and investor state

    Returns:
        np.float64: Excess demand (positive means demand > supply)
    """
    total_demand = 0.0
    annualized_earnings = ts.annualize(mkt, t)
    for investor in ts.investors:
        # Calculate desired equity position for each investor
        match investor.investor_type():
            case "extrapolator":
                n_year_annualized_return = investor.weighted_avg_returns(
                    ts.return_idx, Extrapolator.weights_5_36(), t
                )
                desired_equity = investor.desired_equity(
                    t, n_year_annualized_return, mkt, ts.price_idx[t - 1], price
                )
            case "long_term":
                desired_equity = investor.desired_equity(
                    t, annualized_earnings, ts.price_idx[t - 1], price
                )
            case "rebalancer_60_40":
                desired_equity = investor.desired_equity()
            case "noise":
                desired_equity = noise
            case _:
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
    noise = np.random.uniform(0.4, 0.8)

    def objective(price: float) -> float:
        return market_clearing_error(price, t, ts, mkt, noise)

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
        mkt: Market parameters including simulation length and initial conditions
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
        annualized_earnings = ts.annualize(mkt, t)

        for investor in investors:
            if investor.investor_type() == "extrapolator":
                n_year_annualized_return = investor.weighted_avg_returns(
                    ts.return_idx, Extrapolator.weights_5_36(), t
                )
                investor.expected_return = investor.calculate_expected_return(
                    t, n_year_annualized_return, mkt
                )
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
                    investor.equity()[t] = (
                        investor.wealth()[t]
                        * investor.expected_return
                        / (investor.gamma() * investor.sigma() ** 2)
                    )
                case "long_term":
                    investor.equity()[t] = (
                        investor.wealth()[t]
                        * investor.calculate_expected_return(
                            annualized_earnings,
                            ts.price_idx[t],
                        )
                        / (investor.gamma() * investor.sigma() ** 2)
                    )
                case "rebalancer_60_40":
                    investor.equity()[t] = investor.wealth()[t] * investor.desired_equity()

            investor.cash()[t] = investor.wealth()[t] - investor.equity()[t]

        fair_value = annualized_earnings / mkt.initial_expected_return

    # Create column names based on investor types
    investor_columns = {}

    for i, investor in enumerate(ts.investors):
        inv_type = investor.investor_type()
        investor_columns[f"Wealth {inv_type}"] = investor.wealth()
        investor_columns[f"Equity {inv_type}"] = investor.equity()
        investor_columns[f"Cash {inv_type}"] = investor.cash()
        investor_columns[f"Share {inv_type}"] = investor.equity() / investor.wealth()

    return pl.DataFrame(
        {
            "Month": list(range(len(ts.monthly_earnings))),
            "Return Idx": ts.return_idx,
            "Price Idx": ts.price_idx,
            "Premium": np.log(ts.price_idx / fair_value),
            "Fair Value": fair_value,
            **investor_columns,  # Unpack the investor columns
        }
    )
