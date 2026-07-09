from datetime import datetime, timedelta

import pytest

from src.backtest_engine import BacktestEngine
from strategies.momentum import MomentumCrossoverStrategy


def _integration_bars(count: int = 80) -> dict[str, list[dict]]:
    bars = []
    base = datetime(2020, 1, 1)
    price = 100.0
    for i in range(count):
        wave = 5.0 * (1 if i % 20 < 10 else -1)
        bars.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": price,
                "high": price + 2,
                "low": price - 2,
                "close": price + wave * 0.1,
                "volume": 2_000_000,
                "adj_close": price + wave * 0.1,
            }
        )
        price += wave * 0.05
    return {"SPY": bars}


def test_backtest_engine_runs_end_to_end() -> None:
    engine = BacktestEngine(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-06-01",
        initial_capital=100_000.0,
        strategy_class=MomentumCrossoverStrategy,
        strategy_kwargs={"symbols": ["SPY"], "fast_window": 5, "slow_window": 20},
        bars=_integration_bars(),
    )
    equity = engine.run()
    assert not equity.empty
    assert "total" in equity.columns
    assert "returns" in equity.columns
