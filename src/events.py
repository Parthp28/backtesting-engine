from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"


class OrderDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MKT = "MKT"
    LMT = "LMT"


@dataclass
class MarketEvent:
  # Why: decouples data feed from strategy and portfolio so either side can be swapped.
    event_type: str = "MARKET"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    adj_close: float = 0.0


@dataclass
class SignalEvent:
  # Why: strategy expresses direction only; portfolio owns sizing and risk limits.
    event_type: str = "SIGNAL"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    signal_type: SignalType = SignalType.LONG
    strength: float = 1.0
    strategy_id: str = ""


@dataclass
class OrderEvent:
  # Why: portfolio may reject or resize a signal before anything hits execution.
    event_type: str = "ORDER"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    order_type: OrderType = OrderType.MKT
    quantity: int = 0
    direction: OrderDirection = OrderDirection.BUY


@dataclass
class FillEvent:
  # Why: fill price includes slippage and may differ from the order's reference price.
    event_type: str = "FILL"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    quantity: int = 0
    direction: OrderDirection = OrderDirection.BUY
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    partial: bool = False
    remaining_quantity: int = 0
