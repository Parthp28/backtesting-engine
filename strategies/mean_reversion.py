from collections import deque
from typing import Any

import numpy as np

from src.events import MarketEvent, SignalEvent, SignalType
from strategies.base import Strategy


class MeanReversionStrategy(Strategy):
    """Rolling z-score mean reversion on adj_close."""

    def __init__(
        self,
        data_handler: Any,
        event_queue: deque,
        symbols: list[str],
        z_window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        super().__init__(data_handler, event_queue)
        self.symbols = symbols
        self.z_window = z_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.long: dict[str, bool] = {s: False for s in symbols}
        self.short: dict[str, bool] = {s: False for s in symbols}

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.symbol not in self.symbols:
            return
        symbol = event.symbol
        bars = self.data.get_latest_bars(symbol, self.z_window)
        if len(bars) < self.z_window:
            return

        prices = np.array([bar["adj_close"] for bar in bars])
        std = float(np.std(prices))
        z_score = float((prices[-1] - np.mean(prices)) / std) if std > 0 else 0.0

        if abs(z_score) < self.exit_z and (self.long[symbol] or self.short[symbol]):
            self.long[symbol] = False
            self.short[symbol] = False
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    strategy_id="mean_reversion",
                )
            )
            return

        if z_score < -self.entry_z and not self.long[symbol]:
            self.long[symbol] = True
            self.short[symbol] = False
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strategy_id="mean_reversion",
                )
            )
        elif z_score > self.entry_z and not self.short[symbol]:
            self.short[symbol] = True
            self.long[symbol] = False
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strategy_id="mean_reversion",
                )
            )
