from collections import defaultdict
from typing import Any

import pandas as pd

from src.events import (
    FillEvent,
    MarketEvent,
    OrderDirection,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)


class Portfolio:
    """Tracks positions, cash, and equity. Converts signals to sized orders."""

    def __init__(self, data_handler: Any, initial_capital: float = 100_000.0) -> None:
        self.data = data_handler
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.current_positions: dict[str, int] = defaultdict(int)
        self.current_holdings: dict[str, float] = defaultdict(float)
        self.all_holdings: list[dict[str, Any]] = []
        self.closed_trades: list[dict[str, float]] = []
        self._open_trade_cost: dict[str, float] = defaultdict(float)

    def update_timeindex(self, event: MarketEvent) -> None:
        for symbol in self.data.symbols:
            price = self.data.get_latest_bar_value(symbol, "adj_close") or 0.0
            self.current_holdings[symbol] = self.current_positions[symbol] * price

        total_equity = self.current_cash + sum(self.current_holdings.values())
        snapshot: dict[str, Any] = {
            "timestamp": event.timestamp,
            "cash": self.current_cash,
            "total": total_equity,
        }
        for symbol in self.data.symbols:
            snapshot[symbol] = self.current_holdings[symbol]
        self.all_holdings.append(snapshot)

    def update_signal(self, event: SignalEvent) -> OrderEvent | None:
        current_price = self.data.get_latest_bar_value(event.symbol, "adj_close")
        if not current_price or current_price <= 0:
            return None

        total_equity = self.current_cash + sum(self.current_holdings.values())
        max_position_value = total_equity * 0.10 * event.strength
        quantity = int(max_position_value / current_price)

        if quantity <= 0 and event.signal_type != SignalType.EXIT:
            return None

        if event.signal_type == SignalType.LONG:
            direction = OrderDirection.BUY
        elif event.signal_type == SignalType.SHORT:
            direction = OrderDirection.SELL
        elif event.signal_type == SignalType.EXIT:
            existing = self.current_positions.get(event.symbol, 0)
            if existing == 0:
                return None
            direction = OrderDirection.SELL if existing > 0 else OrderDirection.BUY
            quantity = abs(existing)
        else:
            return None

        return OrderEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            order_type=OrderType.MKT,
            quantity=quantity,
            direction=direction,
        )

    def update_fill(self, event: FillEvent) -> None:
        cost = event.fill_price * event.quantity
        if event.direction == OrderDirection.BUY:
            self.current_positions[event.symbol] += event.quantity
            self.current_cash -= cost + event.commission
            self._open_trade_cost[event.symbol] += cost + event.commission
        else:
            entry_cost = self._open_trade_cost.get(event.symbol, 0.0)
            proceeds = cost - event.commission
            prior_position = self.current_positions[event.symbol]
            self.current_positions[event.symbol] -= event.quantity
            self.current_cash += proceeds
            if prior_position != 0 and self.current_positions[event.symbol] == 0:
                pnl = proceeds - entry_cost
                self.closed_trades.append({"symbol": event.symbol, "pnl": pnl})
                self._open_trade_cost[event.symbol] = 0.0
            elif prior_position != 0:
                closed_fraction = event.quantity / prior_position
                self._open_trade_cost[event.symbol] *= max(0.0, 1.0 - closed_fraction)

    def create_equity_curve(self) -> pd.DataFrame:
        df = pd.DataFrame(self.all_holdings)
        if df.empty:
            return df
        df = df.set_index("timestamp")
        df["returns"] = df["total"].pct_change()
        return df
