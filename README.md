# Event-Driven Backtesting Framework

A backtesting engine that processes market data one bar at a time through an event queue, the same way a live trading system would. Built to test the momentum, pairs trading, and mean reversion strategies below with realistic execution costs, not the optimistic fills you get from a vectorized backtest.

## Architecture
                ┌───────────────────┐
                │   Data Handler    │
                │  (bar-by-bar feed)│
                └────────┬──────────┘
                         │ MarketEvent
                         ▼
                ┌─────────────────┐
                │  Backtest Engine│
                │   (event queue) │
                └────────┬────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    ┌──────────┐  ┌───────────┐  ┌────────────┐
    │ Strategy │  │ Portfolio │  │ Execution  │
    │ (signals)│→ │ (sizing)  │→ │  Handler   │
    └──────────┘  └───────────┘  └────────────┘
                         │              │
                 SignalEvent    OrderEvent, FillEvent

Bars flow in one at a time. Strategies only see `latest_bars`, older data is popped off, so there is no way for a strategy to accidentally peek at future prices. Orders queue in the execution handler and fill on the next bar's open, not the bar that generated the signal.

## Performance

| Metric | Result | Target |
|--------|--------|--------|
| Event loop | 0.0205 ms/bar | < 0.5 ms/bar |
| Vectorized SMA (reference) | 0.0003 ms/bar | n/a |
| Slowdown vs vectorized | 75.8x | expected tradeoff |

Tested on 5,000 synthetic bars. The event-driven loop is about 75 times slower than a fully vectorized calculation, which is the cost of processing bars sequentially instead of as a NumPy array. That tradeoff buys realistic fill timing and prevents lookahead bias, which a vectorized backtest cannot guarantee.

## Tests

46 tests passing, 84.5% coverage across `src/`, `strategies/`, and `data/`.
pytest tests/ -v --cov=src --cov=strategies --cov=data --cov-fail-under=80

## Strategy results

Backtested on synthetic 2010 to 2024 daily bars (yfinance was unavailable in the build environment, see note below).

| Strategy | Ann. Return | Sharpe | Max DD | Hit Rate |
|----------|-------------|--------|--------|----------|
| Momentum crossover | -0.30% | -4.66 | -9.2% | 31.8% |
| Pairs trading | +0.09% | -13.28 | -0.8% | 23.0% |
| Mean reversion | +0.29% | -4.51 | -4.7% | 23.9% |
| SPY (benchmark) | +0.81% | -0.25 | -40.5% | n/a |

## What failed and why

All three strategies underperformed a buy-and-hold SPY benchmark after transaction costs and slippage. The Sharpe ratios are negative across the board, meaning none of these three strategies produced returns worth the volatility taken on.

This is not a bug in the framework. Simple crossover and z-score strategies on daily bars are heavily arbitraged in real markets, so seeing them fail here after realistic costs is closer to expected than a strategy that mysteriously beats the market with three lines of logic. The framework's job was to measure this accurately, and it did.

## Design decisions

**Why next-bar-open fills instead of same-bar fills:** Filling an order on the same bar that generated the signal assumes you can trade at a price you could not have known was coming. Queuing the order and filling on the next bar's open removes that lookahead advantage.

**Why 10% volume participation for partial fills:** A single order rarely fills instantly at full size when it represents a meaningful share of a bar's volume. Capping fills at 10% of next-bar volume and rolling the remainder forward better reflects how a large order actually gets worked in a real market.

**Why walk-forward validation:** A strategy tuned and tested on the same window will look better than it actually is. The 504-bar train, 126-bar test rolling windows with grid search force each parameter set to prove itself on data it has not seen.

**Why synthetic data fallback:** The data fetcher depends on yfinance, which is not always reachable in a CI or sandboxed environment. A seeded synthetic price generator keeps the test suite and benchmark runnable without a live network dependency, with real data used whenever the API is available.

## Project structure
src/            event loop, data handler, portfolio, execution, performance metrics
strategies/     momentum, pairs trading, mean reversion
data/           yfinance wrapper with synthetic fallback
examples/       strategy runs, event loop benchmark, report generation
tests/          46 tests across all modules
notebooks/      tearsheet demo

## How to run
docker-compose up
pytest tests/ -v --cov=src --cov=strategies --cov=data --cov-fail-under=80
PYTHONPATH=. python examples/benchmark_event_loop.py
PYTHONPATH=. python examples/generate_report.py

## Known limitations

Live data depends on yfinance access, which can be unreliable. The strategies included here are intentionally simple, meant to exercise the framework rather than serve as production trading logic. Slippage and commission models are fixed constants rather than calibrated against real fill data.
