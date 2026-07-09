import numpy as np
import pandas as pd
import pytest

from src.performance import (
    compute_hit_rate,
    compute_information_coefficient,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)


def test_sharpe_positive_for_positive_returns() -> None:
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.005])
    assert compute_sharpe_ratio(returns) > 0


def test_sharpe_zero_for_flat_returns() -> None:
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert compute_sharpe_ratio(returns) == 0.0


def test_max_drawdown_negative_value() -> None:
    equity = pd.Series([100, 110, 90, 95])
    max_dd, _, _ = compute_max_drawdown(equity)
    assert max_dd < 0


def test_max_drawdown_correct_on_known_series() -> None:
    equity = pd.Series([100.0, 120.0, 90.0])
    max_dd, _, _ = compute_max_drawdown(equity)
    expected = (90.0 - 120.0) / 120.0
    assert max_dd == pytest.approx(expected)


def test_sortino_better_than_sharpe_for_right_skewed_returns() -> None:
    returns = pd.Series([0.05, 0.04, 0.03, -0.01, -0.005, 0.06, 0.02])
    sharpe = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    assert sortino > sharpe


def test_ic_zero_for_random_signals() -> None:
    rng = np.random.default_rng(0)
    signals = pd.Series(rng.normal(size=200))
    forward = pd.Series(rng.normal(size=200))
    ic = compute_information_coefficient(signals, forward)
    assert abs(ic) < 0.2


def test_ic_positive_for_perfect_predictor() -> None:
    signals = pd.Series([1, 2, 3, 4, 5, 6])
    forward = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ic = compute_information_coefficient(signals, forward)
    assert ic > 0.9


def test_hit_rate_fifty_percent_for_even_split() -> None:
    trades = [{"pnl": 1.0}, {"pnl": -1.0}]
    assert compute_hit_rate(trades) == 0.5


def test_profit_factor_above_one_for_profitable_strategy() -> None:
    trades = [{"pnl": 3.0}, {"pnl": 2.0}, {"pnl": -1.0}]
    assert compute_profit_factor(trades) > 1.0
