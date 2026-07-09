"""Benchmark event-driven loop vs vectorized SMA crossover."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest_engine import BacktestEngine
from strategies.momentum import MomentumCrossoverStrategy

TARGET_EVENT_MS_PER_BAR = 0.5
BAR_COUNT = 5000


def _synthetic_bars(count: int) -> dict[str, list[dict]]:
    bars = []
    base = datetime(2010, 1, 1)
    price = 100.0
    rng = np.random.default_rng(7)
    for i in range(count):
        price += rng.normal(0, 0.5)
        bars.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.1,
                "volume": 5_000_000,
                "adj_close": price + 0.1,
            }
        )
    return {"SPY": bars}


def benchmark_event_driven(bars: dict[str, list[dict]]) -> float:
    engine = BacktestEngine(
        symbols=["SPY"],
        start="2010-01-01",
        end="2025-01-01",
        initial_capital=100_000.0,
        strategy_class=MomentumCrossoverStrategy,
        strategy_kwargs={"symbols": ["SPY"], "fast_window": 10, "slow_window": 50},
        bars=bars,
    )
    start = time.perf_counter()
    engine.run()
    elapsed = time.perf_counter() - start
    return elapsed / BAR_COUNT * 1000.0


def benchmark_vectorized(bars: dict[str, list[dict]]) -> float:
    df = pd.DataFrame(bars["SPY"]).set_index("timestamp")
    start = time.perf_counter()
    fast = df["adj_close"].rolling(10).mean()
    slow = df["adj_close"].rolling(50).mean()
    signal = (fast > slow).astype(int)
    _ = signal.diff()
    elapsed = time.perf_counter() - start
    return elapsed / BAR_COUNT * 1000.0


if __name__ == "__main__":
    bars = _synthetic_bars(BAR_COUNT)
    event_ms = benchmark_event_driven(bars)
    vector_ms = benchmark_vectorized(bars)
    print(f"Bars: {BAR_COUNT}")
    print(f"Event-driven: {event_ms:.4f} ms/bar (target < {TARGET_EVENT_MS_PER_BAR} ms/bar)")
    print(f"Vectorized:   {vector_ms:.4f} ms/bar")
    print(f"Ratio (event/vector): {event_ms / vector_ms:.2f}x")
