from collections import deque
from typing import Any

from data.fetcher import load_bars
from src.events import MarketEvent


class HistoricalDataHandler:
    """Feeds historical data one bar at a time with structural lookahead prevention."""

    def __init__(
        self,
        symbols: list[str],
        start: str,
        end: str,
        bars: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.symbols = symbols
        self.start = start
        self.end = end
        self.all_data: dict[str, deque[dict[str, Any]]] = {}
        self.latest_bars: dict[str, list[dict[str, Any]]] = {}
        self.continue_backtest: bool = True
        self._next_bar_opens: dict[str, float] = {}
        if bars is not None:
            self._load_bars(bars)
        else:
            self._download_data()

    def _load_bars(self, bars: dict[str, list[dict[str, Any]]]) -> None:
        for symbol in self.symbols:
            symbol_bars = bars.get(symbol, [])
            self.all_data[symbol] = deque(symbol_bars)
            self.latest_bars[symbol] = []

    def _download_data(self) -> None:
        for symbol in self.symbols:
            bar_list = load_bars(symbol, self.start, self.end)
            self.all_data[symbol] = deque(bar_list)
            self.latest_bars[symbol] = []

    def update_bars(self) -> MarketEvent | None:
        if any(not self.all_data[symbol] for symbol in self.symbols):
            self.continue_backtest = False
            return None

        new_bar: MarketEvent | None = None
        for symbol in self.symbols:
            bar = self.all_data[symbol].popleft()
            if self.all_data[symbol]:
                self._next_bar_opens[symbol] = float(self.all_data[symbol][0]["open"])
            else:
                self._next_bar_opens.pop(symbol, None)
            self.latest_bars[symbol].append(bar)
            if new_bar is None:
                new_bar = MarketEvent(
                    timestamp=bar["timestamp"],
                    symbol=symbol,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                    close=bar["close"],
                    volume=bar["volume"],
                    adj_close=bar["adj_close"],
                )

        if all(not self.all_data[symbol] for symbol in self.symbols):
            self.continue_backtest = False
        return new_bar

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[dict[str, Any]]:
        bars = self.latest_bars.get(symbol, [])
        return bars[-n:] if len(bars) >= n else []

    def get_latest_bar_value(self, symbol: str, field: str) -> float | None:
        bars = self.get_latest_bars(symbol, 1)
        if not bars:
            return None
        value = bars[-1].get(field)
        return float(value) if value is not None else None

    def get_next_bar_open(self, symbol: str) -> float | None:
        if symbol in self._next_bar_opens:
            return self._next_bar_opens[symbol]
        return None

    def peek_next_bar_volume(self, symbol: str) -> int | None:
        if not self.all_data.get(symbol):
            return None
        return int(self.all_data[symbol][0]["volume"])
