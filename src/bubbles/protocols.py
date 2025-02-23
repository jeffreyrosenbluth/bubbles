from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class InvestorProvider(Protocol):
    """Protocol defining the interface for investor types.

    Methods:
        percent: Returns the investor's percentage of total market
        gamma: Returns the investor's risk aversion parameter
        sigma: Returns the investor's volatility parameter
        merton_share: Calculates optimal portfolio share based on Merton's formula
        expected_return: Returns array of expected returns over time
        wealth: Returns array of total wealth over time
        equity: Returns array of equity holdings over time
        cash: Returns array of cash holdings over time
        cash_post_distribution: Returns array of cash positions after distributions
    """

    def percent(self) -> float: ...
    def gamma(self) -> float: ...
    def sigma(self) -> float: ...

    def merton_share(self, excess_return: float) -> float:
        ...
        # return excess_return / (self.gamma() * self.sigma() ** 2)

    def calculate_expected_return(
        self, t: int, annualized_earnings: float, n_year_annualized_return: float, price: float
    ) -> float: ...

    def wealth(self) -> NDArray[np.float64]: ...
    def equity(self) -> NDArray[np.float64]: ...
    def cash(self) -> NDArray[np.float64]: ...
    def cash_post_distribution(self) -> NDArray[np.float64]: ...
    def expected_return(self) -> NDArray[np.float64]: ...
