from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray


class InvestorProvider(Protocol):
    """Protocol defining the interface for investor types.

    Methods:
        investor_type: Returns the type of investor ('extrapolator' or 'long_term')
        percent: Returns the investor's percentage of total market
        gamma: Returns the investor's  CRRA risk aversion parameter
        sigma: Returns the investor's equity price volatility parameter
        merton_share: Calculates optimal portfolio share based on Merton's formula
        expected_return: Returns array of expected returns over time
        wealth: Returns array of total wealth over time
        equity: Returns array of equity holdings over time
        cash: Returns array of cash holdings over time
        cash_post_distribution: Returns array of cash positions after distributions
    """

    def investor_type(
        self,
    ) -> Literal["extrapolator", "long_term", "rebalancer_60_40", "noise"]: ...
    def percent(self) -> float: ...
    def gamma(self) -> float: ...
    def sigma(self) -> float: ...

    def merton_share(self, excess_return: float) -> float: ...
    def desired_equity(self, *args: any) -> float: ...

    def wealth(self) -> NDArray[np.float64]: ...
    def equity(self) -> NDArray[np.float64]: ...
    def cash(self) -> NDArray[np.float64]: ...
    def cash_post_distribution(self) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class Market:
    """Global parameters for market simulation.

    Attributes:
        years: Number of years to simulate after history
        initial_expected_return: Starting expected return rate
        earnings_vol: Volatility of earnings
        payout_ratio: Ratio of earnings paid as dividends
        history_length: Number of years of history, i.e before the simulation starts
        seed: Random seed for reproducibility (0 uses predefined dz.py values)
    """

    years: int = 50
    initial_expected_return: float = 0.04
    earnings_vol: float = 0.10
    payout_ratio: float = 0.5
    history_length: int = 5
    seed: int = 1337194922  # 0 for dz.py

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
