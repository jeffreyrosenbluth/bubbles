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
        calculate_expected_return: Calculates the expected return for given parameters
        expected_return: Returns array of expected returns over time
        wealth: Returns array of total wealth over time
        equity: Returns array of equity holdings over time
        cash: Returns array of cash holdings over time
        cash_post_distribution: Returns array of cash positions after distributions
    """

    def investor_type(self) -> Literal["extrapolator", "long_term"]: ...
    def percent(self) -> np.float64: ...
    def gamma(self) -> np.float64: ...
    def sigma(self) -> np.float64: ...

    def merton_share(self, excess_return: np.float64) -> np.float64: ...

    def calculate_expected_return(self, *args) -> float: ...

    def wealth(self) -> NDArray[np.float64]: ...
    def equity(self) -> NDArray[np.float64]: ...
    def cash(self) -> NDArray[np.float64]: ...
    def cash_post_distribution(self) -> NDArray[np.float64]: ...
    def expected_return(self) -> NDArray[np.float64]: ...
