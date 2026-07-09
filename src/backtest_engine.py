from collections import deque
from itertools import product
from typing import Any, Type

import pandas as pd

from src.data_handler import HistoricalDataHandler
from src.execution_handler import SimulatedExecutionHandler
from src.performance import compute_sharpe_ratio, generate_summary
from src.portfolio import Portfolio
from strategies.base import Strategy


class BacktestEngine:
    """Event-driven backtest loop wiring data, strategy, portfolio, and execution."""

    def __init__(
        self,
        symbols: list[str],
        start: str,
        end: str,
        initial_capital: float,
        strategy_class: Type[Strategy],
        strategy_kwargs: dict[str, Any] | None = None,
        bars: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.events: deque = deque()
        self.data = HistoricalDataHandler(symbols, start, end, bars=bars)
        kwargs = strategy_kwargs or {}
        self.strategy = strategy_class(self.data, self.events, **kwargs)
        self.portfolio = Portfolio(self.data, initial_capital)
        self.execution = SimulatedExecutionHandler(self.data)

    def run(self) -> pd.DataFrame:
        while self.data.continue_backtest:
            market_event = self.data.update_bars()
            if market_event:
                self.events.append(market_event)

            while self.events:
                event = self.events.popleft()
                if event.event_type == "MARKET":
                    for fill in self.execution.process_pending_orders(event.timestamp):
                        self.events.append(fill)
                    self.strategy.calculate_signals(event)
                    self.portfolio.update_timeindex(event)
                elif event.event_type == "SIGNAL":
                    order = self.portfolio.update_signal(event)
                    if order:
                        self.events.append(order)
                elif event.event_type == "ORDER":
                    self.execution.submit_order(event)
                elif event.event_type == "FILL":
                    self.portfolio.update_fill(event)

        return self.portfolio.create_equity_curve()


class WalkForwardEngine:
    """Walk-forward optimization over rolling train and test windows."""

    def __init__(
        self,
        symbols: list[str],
        start: str,
        end: str,
        strategy_class: Type[Strategy],
        param_grid: dict[str, list[Any]],
        train_window_bars: int = 504,
        test_window_bars: int = 126,
        step_bars: int = 126,
        initial_capital: float = 100_000.0,
        bars: dict[str, list[dict[str, Any]]] | None = None,
        base_strategy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.symbols = symbols
        self.start = start
        self.end = end
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.train_window_bars = train_window_bars
        self.test_window_bars = test_window_bars
        self.step_bars = step_bars
        self.initial_capital = initial_capital
        self.bars = bars
        self.base_strategy_kwargs = base_strategy_kwargs or {}
        self.oos_segments: list[pd.DataFrame] = []
        self.in_sample_segments: list[pd.DataFrame] = []

    def _slice_bars(
        self, bars: dict[str, list[dict[str, Any]]], start_idx: int, end_idx: int
    ) -> dict[str, list[dict[str, Any]]]:
        sliced: dict[str, list[dict[str, Any]]] = {}
        for symbol, symbol_bars in bars.items():
            sliced[symbol] = symbol_bars[start_idx:end_idx]
        return sliced

    def _load_full_bars(self) -> dict[str, list[dict[str, Any]]]:
        if self.bars is not None:
            return self.bars
        loader = HistoricalDataHandler(self.symbols, self.start, self.end)
        return {symbol: list(loader.all_data[symbol]) for symbol in self.symbols}

    def _run_window(
        self, window_bars: dict[str, list[dict[str, Any]]], params: dict[str, Any]
    ) -> pd.DataFrame:
        merged = {**self.base_strategy_kwargs, **params}
        engine = BacktestEngine(
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            initial_capital=self.initial_capital,
            strategy_class=self.strategy_class,
            strategy_kwargs=merged,
            bars=window_bars,
        )
        return engine.run()

    def _best_params(self, train_bars: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        best_params: dict[str, Any] = {}
        best_sharpe = float("-inf")

        for combo in product(*values):
            params = dict(zip(keys, combo))
            equity = self._run_window(train_bars, params)
            if equity.empty or equity["returns"].dropna().empty:
                continue
            sharpe = compute_sharpe_ratio(equity["returns"].dropna())
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        return best_params

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        full_bars = self._load_full_bars()
        bar_count = min(len(full_bars[s]) for s in self.symbols)
        idx = 0

        while idx + self.train_window_bars + self.test_window_bars <= bar_count:
            train_slice = self._slice_bars(full_bars, idx, idx + self.train_window_bars)
            test_slice = self._slice_bars(
                full_bars,
                idx + self.train_window_bars,
                idx + self.train_window_bars + self.test_window_bars,
            )
            best_params = self._best_params(train_slice)
            if not best_params:
                idx += self.step_bars
                continue

            is_equity = self._run_window(train_slice, best_params)
            oos_equity = self._run_window(test_slice, best_params)
            if not is_equity.empty:
                self.in_sample_segments.append(is_equity)
            if not oos_equity.empty:
                self.oos_segments.append(oos_equity)
            idx += self.step_bars

        oos_curve = (
            pd.concat(self.oos_segments).sort_index() if self.oos_segments else pd.DataFrame()
        )
        is_curve = (
            pd.concat(self.in_sample_segments).sort_index()
            if self.in_sample_segments
            else pd.DataFrame()
        )
        return oos_curve, is_curve

    def summarize_oos(self) -> dict[str, float]:
        if not self.oos_segments:
            return {}
        oos_curve = pd.concat(self.oos_segments).sort_index()
        if oos_curve.empty:
            return {}
        return generate_summary(oos_curve)
