from datetime import datetime, timedelta

import pytest

from src.backtest_engine import BacktestEngine, WalkForwardEngine
from src.performance import generate_summary
from strategies.momentum import MomentumCrossoverStrategy


def _bars(count: int) -> dict[str, list[dict]]:
    series = []
    base = datetime(2020, 1, 1)
    price = 100.0
    for i in range(count):
        price += 0.2 if i % 15 < 8 else -0.1
        series.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 3_000_000,
                "adj_close": price,
            }
        )
    return {"SPY": series}


def test_walk_forward_engine_produces_oos_curve() -> None:
    bars = _bars(800)
    wf = WalkForwardEngine(
        symbols=["SPY"],
        start="2020-01-01",
        end="2022-01-01",
        strategy_class=MomentumCrossoverStrategy,
        param_grid={"fast_window": [5, 10], "slow_window": [20, 30]},
        train_window_bars=200,
        test_window_bars=100,
        step_bars=100,
        bars=bars,
        base_strategy_kwargs={"symbols": ["SPY"]},
    )
    oos, is_curve = wf.run()
    assert not oos.empty or not is_curve.empty


def test_generate_summary_on_engine_output() -> None:
    engine = BacktestEngine(
        symbols=["SPY"],
        start="2020-01-01",
        end="2021-01-01",
        initial_capital=100_000.0,
        strategy_class=MomentumCrossoverStrategy,
        strategy_kwargs={"symbols": ["SPY"], "fast_window": 5, "slow_window": 20},
        bars=_bars(120),
    )
    equity = engine.run()
    summary = generate_summary(equity)
    assert "sharpe_ratio" in summary
