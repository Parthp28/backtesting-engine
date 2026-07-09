from collections import deque
from datetime import datetime, timedelta

import pytest

from src.data_handler import HistoricalDataHandler
from src.events import FillEvent, OrderDirection, OrderEvent, OrderType, SignalEvent, SignalType
from src.portfolio import Portfolio


def _handler_with_bar(price: float = 100.0, volume: int = 1_000_000) -> HistoricalDataHandler:
    bars = [
        {
            "timestamp": datetime(2020, 1, 1),
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": volume,
            "adj_close": price,
        }
    ]
    return HistoricalDataHandler(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-01-02",
        bars={"SPY": bars},
    )


def test_initial_cash_equals_initial_capital() -> None:
    handler = _handler_with_bar()
    portfolio = Portfolio(handler, initial_capital=50_000.0)
    assert portfolio.current_cash == 50_000.0


def test_update_fill_buy_reduces_cash() -> None:
    handler = _handler_with_bar()
    portfolio = Portfolio(handler, initial_capital=100_000.0)
    fill = FillEvent(
        symbol="SPY",
        quantity=10,
        direction=OrderDirection.BUY,
        fill_price=100.0,
        commission=1.0,
        slippage=0.1,
    )
    portfolio.update_fill(fill)
    assert portfolio.current_cash == 100_000.0 - 1000.0 - 1.0


def test_update_fill_sell_increases_cash() -> None:
    handler = _handler_with_bar()
    portfolio = Portfolio(handler, initial_capital=100_000.0)
    portfolio.current_positions["SPY"] = 10
    fill = FillEvent(
        symbol="SPY",
        quantity=10,
        direction=OrderDirection.SELL,
        fill_price=100.0,
        commission=1.0,
        slippage=0.1,
    )
    portfolio.update_fill(fill)
    assert portfolio.current_cash == 100_000.0 + 1000.0 - 1.0


def test_position_negative_after_short() -> None:
    handler = _handler_with_bar()
    portfolio = Portfolio(handler)
    fill = FillEvent(
        symbol="SPY",
        quantity=5,
        direction=OrderDirection.SELL,
        fill_price=100.0,
        commission=1.0,
        slippage=0.1,
    )
    portfolio.update_fill(fill)
    assert portfolio.current_positions["SPY"] == -5


def test_equity_includes_cash_and_holdings() -> None:
    handler = _handler_with_bar(price=100.0)
    handler.update_bars()
    portfolio = Portfolio(handler, initial_capital=10_000.0)
    portfolio.current_positions["SPY"] = 10
    from src.events import MarketEvent

    event = MarketEvent(timestamp=datetime(2020, 1, 1), symbol="SPY")
    portfolio.update_timeindex(event)
    snapshot = portfolio.all_holdings[-1]
    assert snapshot["total"] == portfolio.current_cash + 10 * 100.0


def test_signal_long_creates_buy_order() -> None:
    handler = _handler_with_bar(price=50.0)
    handler.update_bars()
    portfolio = Portfolio(handler, initial_capital=100_000.0)
    signal = SignalEvent(symbol="SPY", signal_type=SignalType.LONG, strength=1.0)
    order = portfolio.update_signal(signal)
    assert order is not None
    assert order.direction == OrderDirection.BUY
    assert order.quantity > 0


def test_signal_exit_long_creates_sell_order() -> None:
    handler = _handler_with_bar(price=50.0)
    handler.update_bars()
    portfolio = Portfolio(handler)
    portfolio.current_positions["SPY"] = 20
    signal = SignalEvent(symbol="SPY", signal_type=SignalType.EXIT)
    order = portfolio.update_signal(signal)
    assert order is not None
    assert order.direction == OrderDirection.SELL
    assert order.quantity == 20


def test_signal_with_zero_equity_returns_none() -> None:
    handler = _handler_with_bar(price=0.0)
    handler.update_bars()
    portfolio = Portfolio(handler, initial_capital=0.0)
    signal = SignalEvent(symbol="SPY", signal_type=SignalType.LONG)
    assert portfolio.update_signal(signal) is None


def test_position_sizing_max_ten_percent() -> None:
    handler = _handler_with_bar(price=100.0)
    handler.update_bars()
    portfolio = Portfolio(handler, initial_capital=100_000.0)
    signal = SignalEvent(symbol="SPY", signal_type=SignalType.LONG, strength=1.0)
    order = portfolio.update_signal(signal)
    assert order is not None
    assert order.quantity * 100.0 <= 100_000.0 * 0.10 + 100.0
