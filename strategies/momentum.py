from collections import deque
from typing import Any

import numpy as np

from src.events import MarketEvent, SignalEvent, SignalType
from strategies.base import Strategy


class MomentumCrossoverStrategy(Strategy):
    """SMA crossover momentum strategy with signal-on-cross-only logic."""

    def __init__(
        self,
        data_handler: Any,
        event_queue: deque,
        symbols: list[str],
        fast_window: int = 10,
        slow_window: int = 50,
    ) -> None:
        super().__init__(data_handler, event_queue)
        self.symbols = symbols
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.bought: dict[str, bool] = {s: False for s in symbols}

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.symbol not in self.symbols:
            return
        symbol = event.symbol
        bars = self.data.get_latest_bars(symbol, self.slow_window + 1)
        if len(bars) < self.slow_window + 1:
            return

        closes = [bar["adj_close"] for bar in bars]
        prev_fast = float(np.mean(closes[-(self.fast_window + 1) : -1]))
        curr_fast = float(np.mean(closes[-self.fast_window :]))
        prev_slow = float(np.mean(closes[-(self.slow_window + 1) : -1]))
        curr_slow = float(np.mean(closes[-self.slow_window :]))

        golden_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        death_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        if golden_cross and not self.bought[symbol]:
            self.bought[symbol] = True
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strategy_id="momentum",
                )
            )
        elif death_cross and self.bought[symbol]:
            self.bought[symbol] = False
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    strategy_id="momentum",
                )
            )
