from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import polars as pl


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


def earnings(months: int, r0: float, payout_ratio: float) -> pl.DataFrame:
    monthly_earnings = np.full((months + 1,), np.nan, dtype=float)
    monthly_earnings[1] = (1 + r0) ** (1 / 12) - 1
    reinvested = monthly_earnings[1] * (1 - payout_ratio)
    price = np.ones(months + 1)
    price[1] = price[0] + reinvested
    for t in range(2, months):
        monthly_earnings[t] = monthly_earnings[t - 1] * (1 + reinvested / price[t - 1])
        reinvested = monthly_earnings[t] * (1 - payout_ratio)
        price[t] = price[t - 1] + reinvested
    return pl.DataFrame(
        {
            "monthly_earnings": monthly_earnings,
            "stock_price_idx": price,
        }
    )


def annual_returns(returns: np.ndarray) -> np.ndarray:
    return (
        np.array([np.prod(1 + returns[i - 12 : i]) for i in range(12, len(returns))])
        - 1
    )


def annual_returns_slice(returns: np.ndarray, m: int, gap: int) -> np.ndarray:
    returns = annual_returns(returns)
    last_idx = len(returns) - 1
    start_idx = last_idx - (m - 1) * gap
    indices = np.arange(start_idx, last_idx + 1, gap)
    print(indices)
    return np.clip(indices, 0, last_idx)


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


reference = {
    "month": "B",
    "monthly_earnings": "D",
    "annualized_earnings": "C, E(t)",
    "monthly_distributed": "E",
    "reinvested": "F",
    "stock_total_return_idx": "G",
    "stock_price_idx": "H, P(t)",
    "total_cash": "I",
    "expected_lt_return": "J",
    "expected_return_lookback": "K",
    "weighted_avg_return_lookback": "L",
    "simple_squeeze": "M",
    "squeeze": "N",
    "asset_alloc_return_ex": "O",
    "wealth_lt": "P, W_l(t)",
    "wealth_ex": "Q, W_x(t)",
    "equity_lt": "R, N_l(t)",
    "equity_ex": "S, N_x(t)",
    "equity_total": "T",
    "cash_lt": "U, C_l(t)",
    "cash_ex": "V, C_x(t)",
    "kappa_lt": "W",
    "kappa_ex": "X",
    "cash_after_distribution_lt": "Y",
    "cash_after_distribution_ex": "Z",
    "dz": "AI",
}


@dataclass
class Parameters:
    years: int = 50
    lookback_years: int = 5
    initial_expected_return: float = 0.04
    initial_monthly_return: float = (1 + initial_expected_return) ** (1 / 12) - 1  # BC3
    earnings_vol: float = 0.10  # sigma_e
    monthly_vol: float = earnings_vol / np.sqrt(12)
    expected_equity_vol_ex: float = 0.16  # sigma_x
    expected_equity_vol_lt: float = 0.16  # sigma_l
    payout_ratio: float = 0.5  # pi
    seed: int = 124
    lt_investors: float = 0.50  # N_l(0)
    lt_gamma: float = 3.0
    lt_vol: float = 0.16
    ex_investors: float = 1 - lt_investors  # N_x(0)
    ex_gamma: float = 3.0
    ex_vol: float = 0.16
    risk_free_rate: float = 0.02
    earnings_growth_mean: float = 0.06
    speed_of_adjustment: float = 0.1
    squeeze_percent: float = 0.5
    squeeze_max_deviation: float = 0.04
    squeeze_target: float = 0.04
    squeezing: float = 0.1


def lookback_table(params: Parameters) -> pl.DataFrame:
    months = params.lookback_years * 12 + 1
    monthly_earnings = np.full((months,), np.nan, dtype=float)
    annualized_earnings = np.full((months,), np.nan, dtype=float)
    monthly_distributed = np.full((months,), np.nan, dtype=float)
    reinvested = np.full((months,), np.nan, dtype=float)
    stock_total_return_idx = np.ones(months)
    stock_price_idx = np.ones(months)
    dz = np.zeros(months)

    monthly_earnings[0] = params.initial_monthly_return
    reinvested[0] = monthly_earnings[0] * (1 - params.payout_ratio)
    monthly_distributed[0] = monthly_earnings[0] * params.payout_ratio

    for t in range(1, months):
        # # if t == 1:
        # #     monthly_earnings[t] = params.initial_monthly_return
        # else:
        monthly_earnings[t] = (
            monthly_earnings[t - 1]
            * (1 + reinvested[t - 1] / stock_price_idx[t - 1])
            * (1 + dz[t] * params.monthly_vol)
        )
        monthly_distributed[t] = monthly_earnings[t] * params.payout_ratio
        reinvested[t] = monthly_earnings[t] - monthly_distributed[t]
        stock_price_idx[t] = stock_price_idx[t - 1] + reinvested[t]
        stock_total_return_idx[t] = (
            stock_total_return_idx[t - 1]
            * (stock_price_idx[t] + monthly_distributed[t])
            / stock_price_idx[t - 1]
        )
        annualized_earnings[t] = (
            ((1 + monthly_earnings[t] / stock_price_idx[t - 1]) ** 12 - 1)
        ) * stock_price_idx[t - 1]

    monthly_earnings = np.insert(monthly_earnings, 0, np.nan)[:-1]
    reinvested = np.insert(reinvested, 0, np.nan)[:-1]
    monthly_distributed = np.insert(monthly_distributed, 0, np.nan)[:-1]
    expected_lt_return = annualized_earnings[1:] / stock_price_idx[:-1]
    expected_lt_return = np.insert(expected_lt_return, 0, np.nan)

    df = pl.DataFrame(
        {
            "month": np.arange(months),
            "monthly_earnings": monthly_earnings,
            "annualized_earnings": annualized_earnings,
            "monthly_distributed": monthly_distributed,
            "reinvested": reinvested,
            "stock_total_return_idx": stock_total_return_idx,
            "stock_price_idx": stock_price_idx,
            "expected_lt_return": expected_lt_return,
            "dz": dz,
        }
    )
    return df


def simulation_table(params: Parameters, lookback_df: pl.DataFrame) -> pl.DataFrame:
    months = params.years * 12
    monthly_earnings = np.ones(months)
    annualized_earnings = np.full((months,), np.nan, dtype=float)
    monthly_distributed = np.full((months,), np.nan, dtype=float)
    reinvested = np.full((months,), np.nan, dtype=float)
    stock_total_return_idx = np.ones(months)
    stock_price_idx = np.ones(months)
    dz = np.zeros(months)

    asset_alloc_return_ex = np.full((months,), np.nan, dtype=float)

    monthly_earnings[-1] = lookback_df.get_column("monthly_earnings")[-1]
    reinvested[-1] = lookback_df.get_column("reinvested")[-1]
    stock_price_idx[-1] = lookback_df.get_column("stock_price_idx")[-1]
    stock_total_return_idx[-1] = lookback_df.get_column("stock_total_return_idx")[-1]

    for t in range(months):
        monthly_earnings[t] = (
            monthly_earnings[t - 1]
            * (1 + reinvested[t - 1] / stock_price_idx[t - 1])
            * (1 + dz[t] * params.monthly_vol)
        )
        reinvested[t] = monthly_earnings[t] * (1 - params.payout_ratio)
        monthly_distributed[t] = monthly_earnings[t] * params.payout_ratio
        stock_price_idx[t] = stock_price_idx[t - 1] + reinvested[t]
        stock_total_return_idx[t] = (
            stock_total_return_idx[t - 1]
            * (stock_price_idx[t] + monthly_distributed[t])
            / stock_price_idx[t - 1]
        )
        annualized_earnings[t] = (
            ((1 + monthly_earnings[t] / stock_price_idx[t - 1]) ** 12 - 1)
        ) * stock_price_idx[t - 1]

    expected_lt_return = annualized_earnings[1:] / stock_price_idx[:-1]
    expected_lt_return = np.insert(expected_lt_return, 0, np.nan)

    df = pl.DataFrame(
        {
            "month": np.arange(months),
            "monthly_earnings": monthly_earnings,
            "annualized_earnings": annualized_earnings,
            "monthly_distributed": monthly_distributed,
            "reinvested": reinvested,
            "stock_total_return_idx": stock_total_return_idx,
            "stock_price_idx": stock_price_idx,
            "expected_lt_return": expected_lt_return,
            "dz": dz,
            "asset_alloc_return_ex": asset_alloc_return_ex,
        }
    )
    return df


# new_arr = np.pad(arr, (n_nans, 0), constant_values=np.nan)


def quad(params: Parameters, df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate positive roots of quadratic equations from coefficients in polars DataFrame
    and append them to the dataframe.

    Assumes coefficients are in columns 'a', 'b', and 'c' for ax^2 + bx + c

    Parameters:
    df (polars.DataFrame): Polars DataFrame containing quadratic coefficients

    Returns:
    polars.DataFrame: Original dataframe with new 'positive_root' column
    """

    row = params.lookback_years * 12 + 1

    a = np.pad(
        df.get_column("asset_alloc_return_ex").to_numpy()[row + 1 :]
        * df.get_column("equity_ex").to_numpy()[row:-1]
        / df.get_column("stock_price_idx").to_numpy()[row:-1]
        / (params.ex_gamma * params.expected_equity_vol_ex**2)
        - 1,
        (row, 0),
        constant_values=np.nan,
    )

    b = np.pad(
        df.get_column("annualized_earnings").to_numpy()[row + 1 :]
        * df.get_column("equity_lt").to_numpy()[row:-1]
        / df.get_column("stock_price_idx").to_numpy()[row:-1]
        / (params.lt_gamma * params.expected_equity_vol_lt**2)
        + df.get_column("asset_alloc_return_ex").to_numpy()[row + 1 :]
        * df.get_column("cash_after_distribution_lt").to_numpy()[row + 1 :]
        / (params.ex_gamma * params.ex_vol**2),
        (row, 0),
        constant_values=np.nan,
    )

    c = np.pad(
        df.get_column("annualized_earnings").to_numpy()[row + 1 :]
        * df.get_column("cash_after_distribution_lt").to_numpy()[row + 1 :]
        / (params.lt_gamma * params.lt_vol**2),
        (row, 0),
        constant_values=np.nan,
    )

    discriminant = b**2 - 4 * a * c
    root1 = (-b + np.sqrt(discriminant)) / (2 * a)
    root2 = (-b - np.sqrt(discriminant)) / (2 * a)
    positive_roots = np.where(root1 > 0, root1, root2)

    return df.with_columns(pl.Series(name="positive_root", values=positive_roots))
