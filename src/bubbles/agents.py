"""
An equilibrium 2-agent model in which the interaction between investors with different return
expectations produces a rich market dynamic that includes equity market volatility in excess of earnings volatility,
short-term trends, long-term mean reversion, high trading volume, and bubbles and crashes.
"""

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


def monthly_earnings_next(
    monthly_earnings_prev: float,
    reinvested: float,
    price_idx_pred: float,
    dz: float,
    earnings_vol: float,
) -> float:
    """Calculate the next month's earnings based on previous earnings and market conditions.

    Args:
        monthly_earnings_prev: Previous month's earnings
        reinvested: Amount of earnings reinvested
        price_idx_pred: Predicted price index
        dz: Random shock term
        earnings_vol: Earnings volatility

    Returns:
        float: Next month's earnings
    """
    return (
        monthly_earnings_prev
        * (1 + reinvested / price_idx_pred)
        * (1 + dz * earnings_vol / SQRT_12)
    )


def annualize(monthly_earnings: float, price_idx_prev: float) -> float:
    """Convert monthly earnings to annualized returns.

    Args:
        monthly_earnings: Monthly earnings value
        price_idx_prev: Previous price index

    Returns:
        float: Annualized earnings value
    """
    return ((1 + monthly_earnings / price_idx_prev) ** 12 - 1) * price_idx_prev


def normalize(
    n_year_annualized_return: float,
    initial_expected_return: float,
    squeeze_params: SqueezeParameters,
) -> float:
    """Normalize returns using a hyperbolic tangent transformation.

    Args:
        n_year_annualized_return: The n-year annualized return
        initial_expected_return: Initial expected return
        squeeze_params: Parameters controlling the squeeze transformation

    Returns:
        float: Normalized return value between squeeze target ± max deviation
    """
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
    """Calculate the next return index value.

    Args:
        return_idx_prev: Previous return index
        monthly_earnings: Current monthly earnings
        price_idx_prev: Previous price index
        price_idx: Current price index
        payout_ratio: Ratio of earnings paid out

    Returns:
        float: Next return index value
    """
    return return_idx_prev * ((price_idx + monthly_earnings * payout_ratio) / price_idx_prev)


def merton_share(
    annualized_earnings: float,
    price_idx: float,
    gamma: float,
    sigma: float,
) -> float:
    """Calculate the Merton optimal portfolio share.

    Args:
        annualized_earnings: The annualized earnings
        price_idx: The price index
        gamma: Risk aversion parameter
        sigma: Volatility parameter

    Returns:
        float: The optimal portfolio share according to Merton's formula
    """
    return annualized_earnings / (price_idx * gamma * sigma**2)


def history_ts(
    mp: MarketParameters,
    ip_x: InvestorParameters,
    ip_y: InvestorParameters,
) -> TimeSeries:
    """Generate historical time series data for market initialization.

    This function simulates historical market data to establish initial conditions
    for the main simulation. It creates a baseline of price indices, earnings,
    and investor positions.

    Args:
        mp: Market parameters
        ip_x: Parameters for investor type X (trend follower)
        ip_y: Parameters for investor type Y (fundamentalist)

    Returns:
        TimeSeries: Historical market data and investor positions
    """
    history_months = 12 * mp.history_length
    ts = initialize_time_series(mp)
    investors = [(ip_x, ts.investor_x), (ip_y, ts.investor_y)]

    ts.monthly_earnings[1] = (1 + mp.initial_expected_return) ** (1 / 12) - 1
    reinvested = ts.monthly_earnings[1] * (1 - mp.payout_ratio)
    ts.price_idx[1] = ts.price_idx[0] + reinvested
    ts.annualized_earnings[1] = annualize(ts.monthly_earnings[1], ts.price_idx[0])
    ts.return_idx[1] = return_idx_next(
        ts.return_idx[0],
        ts.monthly_earnings[1],
        ts.price_idx[0],
        ts.price_idx[1],
        mp.payout_ratio,
    )

    for t in range(2, history_months + 1):
        ts.monthly_earnings[t] = monthly_earnings_next(
            ts.monthly_earnings[t - 1],
            reinvested,
            ts.price_idx[t - 1],
            0,
            mp.earnings_vol,
        )
        reinvested = ts.monthly_earnings[t] * (1 - mp.payout_ratio)

        ts.price_idx[t] = ts.price_idx[t - 1] + reinvested
        ts.annualized_earnings[t] = annualize(ts.monthly_earnings[t], ts.price_idx[t - 1])
        ts.return_idx[t] = return_idx_next(
            ts.return_idx[t - 1],
            ts.monthly_earnings[t],
            ts.price_idx[t - 1],
            ts.price_idx[t],
            mp.payout_ratio,
        )

    expected_return = ts.annualized_earnings[history_months] / ts.price_idx[history_months]
    ts.investor_y.expected_return[history_months] = expected_return
    ts.investor_x.expected_return[history_months] = expected_return
    ts.n_year_annualized_return[history_months] = expected_return

    merton_share_y = merton_share(
        ts.annualized_earnings[history_months],
        ts.price_idx[history_months],
        ip_y.gamma,
        ip_y.sigma,
    )

    merton_share_x = merton_share(
        ts.annualized_earnings[history_months],
        ts.price_idx[history_months],
        ip_x.gamma,
        ip_x.sigma,
    )

    starting_wealth = ts.price_idx[history_months] / (
        ip_y.percent * merton_share_y + ip_x.percent * merton_share_x
    )

    ts.total_cash[history_months] = starting_wealth - ts.price_idx[history_months]

    for ip, inv in investors:
        inv.wealth[history_months] = ip.percent * starting_wealth
        inv.equity[history_months] = (
            inv.wealth[history_months]
            * inv.expected_return[history_months]
            / (ip.gamma * ip.sigma**2)
        )
        inv.cash[history_months] = inv.wealth[history_months] - inv.equity[history_months]
    return ts


def calculate_quadratic_coefficients(
    t: int,
    ts: TimeSeries,
    ip_x: InvestorParameters,
    ip_y: InvestorParameters,
) -> tuple[float, float, float]:
    """Calculate coefficients for the quadratic equation determining the next price index.

    Args:
        t: Current time step
        ts: Time series data
        ip_x: Parameters for trend-following investors
        ip_y: Parameters for fundamentalist investors

    Returns:
        tuple[float, float, float]: Coefficients (a, b, c) for the quadratic equation
    """
    a = (
        ts.investor_x.expected_return[t]
        * ts.investor_x.equity[t - 1]
        / (ts.price_idx[t - 1] * ip_x.gamma * ip_x.sigma**2)
        - 1
    )

    b = ts.annualized_earnings[t] * ts.investor_y.equity[t - 1] / (
        ts.price_idx[t - 1] * ip_y.gamma * ip_y.sigma**2
    ) + ts.investor_x.expected_return[t] * ts.investor_x.cash_post_distribution[t] / (
        ip_x.gamma * ip_x.sigma**2
    )

    c = (
        ts.annualized_earnings[t]
        * ts.investor_y.cash_post_distribution[t]
        / (ip_y.gamma * ip_y.sigma**2)
    )

    return a, b, c


def data_table(
    m: MarketParameters,
    investor_params_x: InvestorParameters,
    investor_params_y: InvestorParameters,
    squeeze_params: SqueezeParameters,
) -> pl.DataFrame:
    """Simulate market dynamics and return results as a DataFrame.

    This function performs the main market simulation, modeling the interaction
    between two types of investors (trend followers and fundamentalists) and
    their impact on market prices and returns.

    Args:
        m: Market parameters including simulation length and initial conditions
        investor_params_x: Parameters for trend-following investors
        investor_params_y: Parameters for fundamentalist investors
        squeeze_params: Parameters controlling return normalization

    Returns:
        pl.DataFrame: DataFrame containing simulation results including:
            - Month: Time index
            - Annualized E: Annualized earnings
            - Monthly E: Monthly earnings
            - Return Idx: Return index
            - Price Idx: Price index
            - Premium: Log premium over fair value
            - Various investor-specific metrics (wealth, equity, cash positions)
    """

    ts = history_ts(m, investor_params_x, investor_params_y)
    history_months = 12 * m.history_length
    months = 12 * (m.years + m.history_length)
    np.random.seed(m.seed)

    investors = [(investor_params_x, ts.investor_x), (investor_params_y, ts.investor_y)]

    reinvested = ts.monthly_earnings[history_months] * (1 - m.payout_ratio)

    for t in range(history_months + 1, months + 1):
        ts.monthly_earnings[t] = monthly_earnings_next(
            ts.monthly_earnings[t - 1],
            reinvested,
            ts.price_idx[t - 1],
            ts.dz[t] if m.seed == 0 else np.random.normal(0, 1),
            m.earnings_vol,
        )
        reinvested = ts.monthly_earnings[t] * (1 - m.payout_ratio)
        ts.annualized_earnings[t] = annualize(ts.monthly_earnings[t], ts.price_idx[t - 1])
        ts.total_cash[t] = ts.total_cash[t - 1] + ts.monthly_earnings[t] * m.payout_ratio
        ts.n_year_annualized_return[t] = weighted_avg_returns(ts.return_idx, return_weights(), t)
        ts.squeeze[t] = normalize(
            ts.n_year_annualized_return[t], m.initial_expected_return, squeeze_params
        )
        ts.investor_x.expected_return[t] = (
            ts.squeeze[t] * investor_params_x.speed_of_adjustment
            + (1 - investor_params_x.speed_of_adjustment) * ts.investor_x.expected_return[t - 1]
        )

        for _, investor in investors:
            investor.cash_post_distribution[t] = (
                investor.cash[t - 1]
                + (ts.monthly_earnings[t] * m.payout_ratio)
                * investor.equity[t - 1]
                / ts.price_idx[t - 1]
            )

        ts.a[t], ts.b[t], ts.c[t] = calculate_quadratic_coefficients(
            t, ts, investor_params_x, investor_params_y
        )

        ts.price_idx[t] = quadratic(ts.a[t], ts.b[t], ts.c[t])
        ts.investor_y.expected_return[t] = ts.annualized_earnings[t] / ts.price_idx[t]
        ts.return_idx[t] = ts.return_idx[t - 1] * (
            (ts.price_idx[t] + ts.monthly_earnings[t] * m.payout_ratio) / ts.price_idx[t - 1]
        )
        for investor_params, investor in investors:
            investor.wealth[t] = (
                investor.cash[t - 1]
                + investor.equity[t - 1] * ts.return_idx[t] / ts.return_idx[t - 1]
            )
            investor.equity[t] = (
                investor.wealth[t]
                * investor.expected_return[t]
                / (investor_params.gamma * investor_params.sigma**2)
            )
            investor.cash[t] = investor.wealth[t] - investor.equity[t]

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
            # "Total_Cash": ts.total_cash,
            "Expected Return y": ts.investor_y.expected_return,
            "Expected Return x": ts.investor_x.expected_return,
            # "n Year Ann Ret": ts.n_year_annualized_return,
            "Wealth x": ts.investor_x.wealth,
            "Wealth y": ts.investor_y.wealth,
            "Equity x": ts.investor_x.equity,
            "Equity y": ts.investor_y.equity,
            "Cash x": ts.investor_x.cash,
            "Cash y": ts.investor_y.cash,
            "Kappa_x": ts.investor_x.equity / ts.investor_x.wealth,
            "Kappa_y": ts.investor_y.equity / ts.investor_y.wealth,
            # # "Cash Post x": ts.investor_x.cash_post_distribution,
            # "Cash Post y": ts.investor_y.cash_post_distribution,
            "Fair Value": fair_value,
        }
    )
