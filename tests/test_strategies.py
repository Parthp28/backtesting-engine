from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

from src.data_handler import HistoricalDataHandler
from src.events import MarketEvent, SignalEvent, SignalType
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumCrossoverStrategy
from strategies.pairs_trading import PairsTradingStrategy


def _make_trend_bars(count: int, start: float, step: float) -> list[dict]:
    bars = []
    base = datetime(2020, 1, 1)
    price = start
    for i in range(count):
        bars.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 1_000_000,
                "adj_close": price + 0.5,
            }
        )
        price += step
    return bars


def _golden_cross_bars(slow_window: int = 5, fast_window: int = 2) -> list[dict]:
    base = datetime(2020, 1, 1)
    flat_count = slow_window + 2
    bars = []
    for i in range(flat_count):
        bars.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000,
                "adj_close": 100.0,
            }
        )
    for j in range(slow_window + 3):
        price = 100.0 + (j + 1) * 5.0
        bars.append(
            {
                "timestamp": base + timedelta(days=flat_count + j),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1_000_000,
                "adj_close": price,
            }
        )
    return bars


def _death_cross_bars(slow_window: int = 5, fast_window: int = 2) -> list[dict]:
    rise = _golden_cross_bars(slow_window, fast_window)
    base_ts = rise[-1]["timestamp"]
    bars = list(rise)
    for j in range(slow_window + 3):
        price = rise[-1]["adj_close"] - (j + 1) * 5.0
        bars.append(
            {
                "timestamp": base_ts + timedelta(days=j + 1),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1_000_000,
                "adj_close": price,
            }
        )
    return bars


@pytest.fixture
def momentum_setup() -> tuple[HistoricalDataHandler, deque, MomentumCrossoverStrategy]:
    bars = _golden_cross_bars(slow_window=5)
    handler = HistoricalDataHandler(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-03-01",
        bars={"SPY": bars},
    )
    queue: deque = deque()
    strategy = MomentumCrossoverStrategy(
        handler, queue, symbols=["SPY"], fast_window=2, slow_window=5
    )
    return handler, queue, strategy


def test_momentum_no_signal_before_slow_window(
    momentum_setup: tuple[HistoricalDataHandler, deque, MomentumCrossoverStrategy],
) -> None:
    handler, queue, strategy = momentum_setup
    for _ in range(5):
        event = handler.update_bars()
        assert event is not None
        strategy.calculate_signals(event)
    assert len(queue) == 0


def test_momentum_long_signal_on_golden_cross(
    momentum_setup: tuple[HistoricalDataHandler, deque, MomentumCrossoverStrategy],
) -> None:
    handler, queue, strategy = momentum_setup
    event = None
    for _ in range(20):
        event = handler.update_bars()
        if event:
            strategy.calculate_signals(event)
    signals = [e for e in queue if isinstance(e, SignalEvent)]
    assert any(s.signal_type == SignalType.LONG for s in signals)


def test_momentum_exit_signal_on_death_cross() -> None:
    bars = _death_cross_bars(slow_window=5)
    handler = HistoricalDataHandler(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-03-01",
        bars={"SPY": bars},
    )
    queue: deque = deque()
    strategy = MomentumCrossoverStrategy(
        handler, queue, symbols=["SPY"], fast_window=2, slow_window=5
    )
    for _ in range(len(bars)):
        event = handler.update_bars()
        if event:
            strategy.calculate_signals(event)
    signals = [e for e in queue if isinstance(e, SignalEvent)]
    assert any(s.signal_type == SignalType.EXIT for s in signals)


def test_momentum_no_signal_when_already_long(
    momentum_setup: tuple[HistoricalDataHandler, deque, MomentumCrossoverStrategy],
) -> None:
    handler, queue, strategy = momentum_setup
    strategy.bought["SPY"] = True
    for _ in range(20):
        event = handler.update_bars()
        if event:
            strategy.calculate_signals(event)
    long_signals = [
        e for e in queue if isinstance(e, SignalEvent) and e.signal_type == SignalType.LONG
    ]
    assert len(long_signals) == 0


def test_pairs_rejects_non_cointegrated_pair() -> None:
    rng = np.random.default_rng(42)
    bars_a = []
    bars_b = []
    base = datetime(2020, 1, 1)
    for i in range(300):
        bars_a.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": float(i),
                "high": float(i + 1),
                "low": float(i - 1),
                "close": float(i),
                "volume": 1_000_000,
                "adj_close": float(i),
            }
        )
        bars_b.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": float(rng.normal()),
                "high": float(rng.normal() + 1),
                "low": float(rng.normal() - 1),
                "close": float(rng.normal()),
                "volume": 1_000_000,
                "adj_close": float(rng.normal()),
            }
        )
    handler = HistoricalDataHandler(
        symbols=["AAA", "BBB"],
        start="2020-01-01",
        end="2021-01-01",
        bars={"AAA": bars_a, "BBB": bars_b},
    )
    for _ in range(252):
        handler.update_bars()
    queue: deque = deque()
    strategy = PairsTradingStrategy(handler, queue, pair=("AAA", "BBB"))
    with pytest.raises(ValueError, match="not cointegrated"):
        for _ in range(300):
            event = handler.update_bars()
            if event:
                strategy.calculate_signals(event)


def _cointegrated_pair_bars(count: int = 80) -> tuple[list[dict], list[dict]]:
    bars_a = []
    bars_b = []
    base = datetime(2020, 1, 1)
    b_price = 50.0
    for i in range(count):
        noise = np.sin(i / 5.0) * 0.5
        a_price = 2.0 * b_price + noise
        bars_a.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": a_price,
                "high": a_price + 1,
                "low": a_price - 1,
                "close": a_price,
                "volume": 1_000_000,
                "adj_close": a_price,
            }
        )
        bars_b.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": b_price,
                "high": b_price + 1,
                "low": b_price - 1,
                "close": b_price,
                "volume": 1_000_000,
                "adj_close": b_price,
            }
        )
        b_price += 0.1
    return bars_a, bars_b


def test_pairs_long_signal_when_z_below_negative_entry() -> None:
    bars_a, bars_b = _cointegrated_pair_bars(count=80)
    bars_a[-1]["adj_close"] -= 25.0
    bars_a[-1]["close"] -= 25.0
    handler = HistoricalDataHandler(
        symbols=["GLD", "SLV"],
        start="2020-01-01",
        end="2020-06-01",
        bars={"GLD": bars_a, "SLV": bars_b},
    )
    queue: deque = deque()
    strategy = PairsTradingStrategy(
        handler,
        queue,
        pair=("GLD", "SLV"),
        skip_cointegration_check=True,
        entry_z=1.0,
        hedge_window=30,
        spread_window=10,
    )
    for _ in range(len(bars_a)):
        event = handler.update_bars()
        if event:
            strategy.calculate_signals(event)
    signals = [e for e in queue if isinstance(e, SignalEvent)]
    assert any(s.signal_type == SignalType.LONG and s.symbol == "GLD" for s in signals)


def test_pairs_exit_signal_when_z_reverts() -> None:
    bars_a, bars_b = _cointegrated_pair_bars(count=80)
    handler = HistoricalDataHandler(
        symbols=["GLD", "SLV"],
        start="2020-01-01",
        end="2020-06-01",
        bars={"GLD": bars_a, "SLV": bars_b},
    )
    queue: deque = deque()
    strategy = PairsTradingStrategy(
        handler,
        queue,
        pair=("GLD", "SLV"),
        skip_cointegration_check=True,
        hedge_window=30,
        spread_window=10,
    )
    for _ in range(60):
        handler.update_bars()
    strategy.in_position = True
    for _ in range(20):
        event = handler.update_bars()
        if event:
            strategy.calculate_signals(event)
    exit_signals = [
        e for e in queue if isinstance(e, SignalEvent) and e.signal_type == SignalType.EXIT
    ]
    assert len(exit_signals) >= 2


def test_pairs_stop_loss_at_z_three() -> None:
    bars_a, bars_b = _cointegrated_pair_bars(count=80)
    bars_a[-1]["adj_close"] -= 120.0
    bars_a[-1]["close"] -= 120.0
    handler = HistoricalDataHandler(
        symbols=["GLD", "SLV"],
        start="2020-01-01",
        end="2020-06-01",
        bars={"GLD": bars_a, "SLV": bars_b},
    )
    queue: deque = deque()
    strategy = PairsTradingStrategy(
        handler,
        queue,
        pair=("GLD", "SLV"),
        skip_cointegration_check=True,
        stop_z=2.9,
        hedge_window=30,
        spread_window=10,
    )
    for _ in range(len(bars_a) - 1):
        handler.update_bars()
    strategy.in_position = True
    event = handler.update_bars()
    if event:
        strategy.calculate_signals(event)
    exit_signals = [
        e for e in queue if isinstance(e, SignalEvent) and e.signal_type == SignalType.EXIT
    ]
    assert len(exit_signals) >= 2


def test_mean_reversion_long_when_z_below_negative_two() -> None:
    bars = _make_trend_bars(25, 100.0, 0.0)
    bars[-1]["adj_close"] = 80.0
    bars[-1]["close"] = 80.0
    handler = HistoricalDataHandler(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-02-01",
        bars={"SPY": bars},
    )
    queue: deque = deque()
    strategy = MeanReversionStrategy(handler, queue, symbols=["SPY"], z_window=20, entry_z=2.0)
    for _ in range(len(bars)):
        event = handler.update_bars()
        if event:
            strategy.calculate_signals(event)
    signals = [e for e in queue if isinstance(e, SignalEvent)]
    assert any(s.signal_type == SignalType.LONG for s in signals)


def test_mean_reversion_exit_when_z_reverts_to_zero() -> None:
    bars = _make_trend_bars(25, 100.0, 0.0)
    handler = HistoricalDataHandler(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-02-01",
        bars={"SPY": bars},
    )
    queue: deque = deque()
    strategy = MeanReversionStrategy(handler, queue, symbols=["SPY"], z_window=20)
    strategy.long["SPY"] = True
    for _ in range(len(bars)):
        event = handler.update_bars()
        if event:
            strategy.calculate_signals(event)
    exit_signals = [
        e for e in queue if isinstance(e, SignalEvent) and e.signal_type == SignalType.EXIT
    ]
    assert len(exit_signals) >= 1
