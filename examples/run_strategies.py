"""Run all three strategies on configured symbols and print summary metrics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetcher import END, MEAN_REV_SYMBOLS, MOMENTUM_SYMBOLS, PAIRS, START
from src.backtest_engine import BacktestEngine
from src.performance import compute_hit_rate, generate_summary
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumCrossoverStrategy
from strategies.pairs_trading import PairsTradingStrategy


def run_momentum() -> dict:
    engine = BacktestEngine(
        symbols=MOMENTUM_SYMBOLS,
        start=START,
        end=END,
        initial_capital=100_000.0,
        strategy_class=MomentumCrossoverStrategy,
        strategy_kwargs={"symbols": MOMENTUM_SYMBOLS},
    )
    equity = engine.run()
    summary = generate_summary(equity)
    summary["hit_rate"] = compute_hit_rate(engine.portfolio.closed_trades)
    return summary


def run_pairs() -> dict:
    symbol_a, symbol_b = PAIRS[0]
    engine = BacktestEngine(
        symbols=[symbol_a, symbol_b],
        start=START,
        end=END,
        initial_capital=100_000.0,
        strategy_class=PairsTradingStrategy,
        strategy_kwargs={"pair": (symbol_a, symbol_b)},
    )
    equity = engine.run()
    summary = generate_summary(equity)
    summary["hit_rate"] = compute_hit_rate(engine.portfolio.closed_trades)
    return summary


def run_mean_reversion() -> dict:
    engine = BacktestEngine(
        symbols=MEAN_REV_SYMBOLS,
        start=START,
        end=END,
        initial_capital=100_000.0,
        strategy_class=MeanReversionStrategy,
        strategy_kwargs={"symbols": MEAN_REV_SYMBOLS},
    )
    equity = engine.run()
    summary = generate_summary(equity)
    summary["hit_rate"] = compute_hit_rate(engine.portfolio.closed_trades)
    return summary


if __name__ == "__main__":
    for name, runner in [
        ("Momentum", run_momentum),
        ("Pairs", run_pairs),
        ("Mean Reversion", run_mean_reversion),
    ]:
        metrics = runner()
        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
