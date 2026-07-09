from datetime import datetime, timedelta

import pytest

from src.data_handler import HistoricalDataHandler


def _make_bars(count: int, start_price: float = 100.0) -> list[dict]:
    bars = []
    base = datetime(2020, 1, 1)
    for i in range(count):
        price = start_price + i
        bars.append(
            {
                "timestamp": base + timedelta(days=i),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 1_000_000,
                "adj_close": price + 0.5,
            }
        )
    return bars


@pytest.fixture
def dual_symbol_handler() -> HistoricalDataHandler:
    bars_a = _make_bars(10, 100.0)
    bars_b = _make_bars(10, 50.0)
    return HistoricalDataHandler(
        symbols=["AAA", "BBB"],
        start="2020-01-01",
        end="2020-01-15",
        bars={"AAA": bars_a, "BBB": bars_b},
    )


def test_update_bars_returns_market_event(dual_symbol_handler: HistoricalDataHandler) -> None:
    event = dual_symbol_handler.update_bars()
    assert event is not None
    assert event.event_type == "MARKET"
    assert event.symbol == "AAA"


def test_update_bars_no_lookahead_all_data_not_accessible(
    dual_symbol_handler: HistoricalDataHandler,
) -> None:
    assert dual_symbol_handler.get_latest_bars("AAA", 1) == []
    dual_symbol_handler.update_bars()
    visible = dual_symbol_handler.get_latest_bars("AAA", 1)
    assert len(visible) == 1
    assert visible[0]["close"] == 100.5


def test_get_latest_bars_returns_n_bars(dual_symbol_handler: HistoricalDataHandler) -> None:
    dual_symbol_handler.update_bars()
    dual_symbol_handler.update_bars()
    bars = dual_symbol_handler.get_latest_bars("AAA", 2)
    assert len(bars) == 2


def test_get_latest_bars_before_n_bars_available_returns_empty(
    dual_symbol_handler: HistoricalDataHandler,
) -> None:
    dual_symbol_handler.update_bars()
    bars = dual_symbol_handler.get_latest_bars("AAA", 5)
    assert bars == []


def test_continue_backtest_false_when_data_exhausted(
    dual_symbol_handler: HistoricalDataHandler,
) -> None:
    for _ in range(10):
        dual_symbol_handler.update_bars()
    assert dual_symbol_handler.continue_backtest is False
    assert dual_symbol_handler.update_bars() is None


def test_all_symbols_advance_simultaneously(dual_symbol_handler: HistoricalDataHandler) -> None:
    dual_symbol_handler.update_bars()
    assert len(dual_symbol_handler.get_latest_bars("AAA", 1)) == 1
    assert len(dual_symbol_handler.get_latest_bars("BBB", 1)) == 1
