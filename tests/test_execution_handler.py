from datetime import datetime

import pytest

from src.data_handler import HistoricalDataHandler
from src.events import OrderDirection, OrderEvent, OrderType
from src.execution_handler import SimulatedExecutionHandler


def _handler_with_next_bar() -> HistoricalDataHandler:
    bars = [
        {
            "timestamp": datetime(2020, 1, 1),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10_000,
            "adj_close": 100.5,
        },
        {
            "timestamp": datetime(2020, 1, 2),
            "open": 102.0,
            "high": 103.0,
            "low": 101.0,
            "close": 102.5,
            "volume": 10_000,
            "adj_close": 102.5,
        },
    ]
    handler = HistoricalDataHandler(
        symbols=["SPY"],
        start="2020-01-01",
        end="2020-01-05",
        bars={"SPY": bars},
    )
    handler.update_bars()
    return handler


def test_buy_fill_price_above_open_by_slippage() -> None:
    handler = _handler_with_next_bar()
    execution = SimulatedExecutionHandler(handler)
    order = OrderEvent(
        symbol="SPY",
        order_type=OrderType.MKT,
        quantity=100,
        direction=OrderDirection.BUY,
    )
    execution.submit_order(order)
    fills = execution.process_pending_orders(datetime(2020, 1, 2))
    assert len(fills) == 1
    next_open = 102.0
    expected = next_open * (1 + SimulatedExecutionHandler.SLIPPAGE_PCT)
    assert fills[0].fill_price == pytest.approx(expected)


def test_sell_fill_price_below_open_by_slippage() -> None:
    handler = _handler_with_next_bar()
    execution = SimulatedExecutionHandler(handler)
    order = OrderEvent(
        symbol="SPY",
        order_type=OrderType.MKT,
        quantity=100,
        direction=OrderDirection.SELL,
    )
    execution.submit_order(order)
    fills = execution.process_pending_orders(datetime(2020, 1, 2))
    next_open = 102.0
    expected = next_open * (1 - SimulatedExecutionHandler.SLIPPAGE_PCT)
    assert fills[0].fill_price == pytest.approx(expected)


def test_commission_minimum_one_dollar() -> None:
    handler = _handler_with_next_bar()
    execution = SimulatedExecutionHandler(handler)
    order = OrderEvent(
        symbol="SPY",
        order_type=OrderType.MKT,
        quantity=10,
        direction=OrderDirection.BUY,
    )
    execution.submit_order(order)
    fills = execution.process_pending_orders(datetime(2020, 1, 2))
    assert fills[0].commission >= 1.0


def test_commission_scales_with_quantity_above_minimum() -> None:
    handler = _handler_with_next_bar()
    execution = SimulatedExecutionHandler(handler)
    order = OrderEvent(
        symbol="SPY",
        order_type=OrderType.MKT,
        quantity=500,
        direction=OrderDirection.BUY,
    )
    execution.submit_order(order)
    fills = execution.process_pending_orders(datetime(2020, 1, 2))
    assert fills[0].commission == pytest.approx(2.5)


def test_partial_fill_when_order_exceeds_volume_participation() -> None:
    handler = _handler_with_next_bar()
    execution = SimulatedExecutionHandler(handler)
    order = OrderEvent(
        symbol="SPY",
        order_type=OrderType.MKT,
        quantity=5000,
        direction=OrderDirection.BUY,
    )
    execution.submit_order(order)
    fills = execution.process_pending_orders(datetime(2020, 1, 2))
    assert fills[0].partial is True
    assert fills[0].quantity == 1000
    assert fills[0].remaining_quantity == 4000
