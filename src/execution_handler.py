from collections import deque
from typing import Any

from src.events import FillEvent, OrderDirection, OrderEvent


class SimulatedExecutionHandler:
    """Fills orders at next bar open with slippage, commission, and partial fills."""

    SLIPPAGE_PCT = 0.001
    PARTICIPATION_RATE = 0.10
    MIN_COMMISSION = 1.0
    COMMISSION_PER_SHARE = 0.005

    def __init__(self, data_handler: Any) -> None:
        self.data = data_handler
        self.pending_orders: deque[OrderEvent] = deque()

    def submit_order(self, event: OrderEvent) -> None:
        self.pending_orders.append(event)

    def process_pending_orders(self, timestamp: Any) -> list[FillEvent]:
        fills: list[FillEvent] = []
        remaining_queue: deque[OrderEvent] = deque()

        while self.pending_orders:
            order = self.pending_orders.popleft()
            fill_events = self._fill_order(order, timestamp)
            if not fill_events:
                continue
            for fill in fill_events:
                fills.append(fill)
                if fill.partial and fill.remaining_quantity > 0:
                    remaining_queue.append(
                        OrderEvent(
                            timestamp=timestamp,
                            symbol=order.symbol,
                            order_type=order.order_type,
                            quantity=fill.remaining_quantity,
                            direction=order.direction,
                        )
                    )

        self.pending_orders.extend(remaining_queue)
        return fills

    def execute_order(self, event: OrderEvent) -> FillEvent | None:
      # Why: defer to next bar so fills never use the signal bar's close.
        self.submit_order(event)
        return None

    def _fill_order(self, event: OrderEvent, timestamp: Any) -> list[FillEvent]:
        next_open = self.data.get_next_bar_open(event.symbol)
        if next_open is None:
            next_open = self.data.get_latest_bar_value(event.symbol, "open")
        if next_open is None:
            return []

        volume = self.data.peek_next_bar_volume(event.symbol)
        if volume is None:
            volume = self.data.get_latest_bar_value(event.symbol, "volume") or 0

        max_fill_qty = max(1, int(volume * self.PARTICIPATION_RATE))
        fill_qty = min(abs(event.quantity), max_fill_qty)
        remaining = abs(event.quantity) - fill_qty
        partial = remaining > 0

        if event.direction == OrderDirection.BUY:
            fill_price = next_open * (1 + self.SLIPPAGE_PCT)
        else:
            fill_price = next_open * (1 - self.SLIPPAGE_PCT)

        commission = max(self.MIN_COMMISSION, self.COMMISSION_PER_SHARE * fill_qty)
        slippage = abs(fill_price - next_open)

        fill = FillEvent(
            timestamp=timestamp,
            symbol=event.symbol,
            quantity=fill_qty,
            direction=event.direction,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            partial=partial,
            remaining_quantity=remaining,
        )
        return [fill]
