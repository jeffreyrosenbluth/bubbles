from dataclasses import dataclass


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
