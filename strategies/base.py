from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from src.events import MarketEvent


class Strategy(ABC):
    """Abstract strategy base. Subclasses emit SignalEvents from MarketEvents."""

    def __init__(self, data_handler: Any, event_queue: deque) -> None:
        self.data = data_handler
        self.queue = event_queue

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        raise NotImplementedError
