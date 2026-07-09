from collections import deque
from typing import Any

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint

from src.events import MarketEvent, SignalEvent, SignalType
from strategies.base import Strategy


class PairsTradingStrategy(Strategy):
    """Cointegrated pairs trading with rolling hedge ratio and z-score signals."""

    def __init__(
        self,
        data_handler: Any,
        event_queue: deque,
        pair: tuple[str, str],
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.0,
        spread_window: int = 20,
        hedge_window: int = 60,
        skip_cointegration_check: bool = False,
    ) -> None:
        super().__init__(data_handler, event_queue)
        self.symbol_a, self.symbol_b = pair
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.spread_window = spread_window
        self.hedge_window = hedge_window
        self.in_position = False
        self.position_side: str | None = None
        self._cointegration_checked = skip_cointegration_check
        self.skip_cointegration_check = skip_cointegration_check

    def _validate_cointegration(self) -> None:
        bars_a = self.data.get_latest_bars(self.symbol_a, 252)
        bars_b = self.data.get_latest_bars(self.symbol_b, 252)
        if len(bars_a) < 252 or len(bars_b) < 252:
            return
        prices_a = np.array([b["adj_close"] for b in bars_a])
        prices_b = np.array([b["adj_close"] for b in bars_b])
        _, p_value, _ = coint(prices_a, prices_b)
        if p_value > 0.05:
            raise ValueError(
                f"Pair {self.symbol_a}/{self.symbol_b} not cointegrated (p={p_value:.3f})"
            )

    def _compute_z_score(self) -> float | None:
        bars_a = self.data.get_latest_bars(self.symbol_a, self.hedge_window)
        bars_b = self.data.get_latest_bars(self.symbol_b, self.hedge_window)
        if len(bars_a) < self.hedge_window or len(bars_b) < self.hedge_window:
            return None

        prices_a = np.array([b["adj_close"] for b in bars_a])
        prices_b = np.array([b["adj_close"] for b in bars_b])
        hedge_ratio = float(OLS(prices_a, prices_b).fit().params[0])

        spread_bars_a = self.data.get_latest_bars(self.symbol_a, self.spread_window)
        spread_bars_b = self.data.get_latest_bars(self.symbol_b, self.spread_window)
        if len(spread_bars_a) < self.spread_window or len(spread_bars_b) < self.spread_window:
            return None

        spread_a = np.array([b["adj_close"] for b in spread_bars_a])
        spread_b = np.array([b["adj_close"] for b in spread_bars_b])
        spread = spread_a - hedge_ratio * spread_b
        std = float(np.std(spread))
        if std == 0:
            return 0.0
        return float((spread[-1] - np.mean(spread)) / std)

    def calculate_signals(self, event: MarketEvent) -> None:
        if event.symbol != self.symbol_a:
            return
        if not self._cointegration_checked and not self.skip_cointegration_check:
            bars_a = self.data.get_latest_bars(self.symbol_a, 252)
            if len(bars_a) < 252:
                return
            self._validate_cointegration()
            self._cointegration_checked = True
        z_score = self._compute_z_score()
        if z_score is None:
            return

        if self.in_position and abs(z_score) > self.stop_z:
            self._emit_exit(event.timestamp)
            return

        if self.in_position and abs(z_score) < self.exit_z:
            self._emit_exit(event.timestamp)
            return

        if not self.in_position and z_score < -self.entry_z:
            self.in_position = True
            self.position_side = "long_spread"
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=self.symbol_a,
                    signal_type=SignalType.LONG,
                    strategy_id="pairs",
                )
            )
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=self.symbol_b,
                    signal_type=SignalType.SHORT,
                    strategy_id="pairs",
                )
            )
        elif not self.in_position and z_score > self.entry_z:
            self.in_position = True
            self.position_side = "short_spread"
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=self.symbol_a,
                    signal_type=SignalType.SHORT,
                    strategy_id="pairs",
                )
            )
            self.queue.append(
                SignalEvent(
                    timestamp=event.timestamp,
                    symbol=self.symbol_b,
                    signal_type=SignalType.LONG,
                    strategy_id="pairs",
                )
            )

    def _emit_exit(self, timestamp: Any) -> None:
        self.in_position = False
        self.position_side = None
        for symbol in (self.symbol_a, self.symbol_b):
            self.queue.append(
                SignalEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    strategy_id="pairs",
                )
            )
