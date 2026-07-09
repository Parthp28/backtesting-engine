from datetime import datetime

import pytest

from src.events import (
    FillEvent,
    MarketEvent,
    OrderDirection,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)


def test_market_event_adj_close_field_exists() -> None:
    event = MarketEvent(symbol="SPY", adj_close=450.25)
    assert event.adj_close == 450.25
    assert event.event_type == "MARKET"


def test_signal_event_strength_between_zero_and_one() -> None:
    event = SignalEvent(symbol="SPY", signal_type=SignalType.LONG, strength=0.75)
    assert 0.0 <= event.strength <= 1.0


def test_order_event_quantity_positive() -> None:
    event = OrderEvent(
        symbol="SPY",
        order_type=OrderType.MKT,
        quantity=100,
        direction=OrderDirection.BUY,
    )
    assert event.quantity > 0


def test_fill_event_commission_at_least_one_dollar() -> None:
    commission = max(1.0, 0.005 * abs(50))
    event = FillEvent(
        symbol="SPY",
        quantity=50,
        direction=OrderDirection.BUY,
        fill_price=100.0,
        commission=commission,
        slippage=0.1,
    )
    assert event.commission >= 1.0
