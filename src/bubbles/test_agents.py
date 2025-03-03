from bubbles.simulation import wealth


def test_wealth():
    w = wealth(100, 0.5, 50)
    assert w == 100
