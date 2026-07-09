"""Generate performance table and walk-forward chart for README."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fetcher import (
    END,
    MEAN_REV_SYMBOLS,
    MOMENTUM_SYMBOLS,
    PAIRS,
    START,
    generate_cointegrated_pair,
    generate_synthetic_bars,
    load_bars,
)
from src.backtest_engine import BacktestEngine, WalkForwardEngine
from src.performance import (
    compute_hit_rate,
    compute_information_coefficient,
    generate_summary,
)
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumCrossoverStrategy
from strategies.pairs_trading import PairsTradingStrategy

OUTPUT_DIR = ROOT / "examples" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_strategy(name: str, engine: BacktestEngine) -> dict[str, float]:
    equity = engine.run()
    summary = generate_summary(equity)
    summary["hit_rate"] = compute_hit_rate(engine.portfolio.closed_trades)
    signals = equity["returns"].fillna(0)
    summary["ic"] = compute_information_coefficient(
        signals.shift(1).fillna(0), signals
    )
    equity["total"].plot(title=f"{name} equity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_equity.png")
    plt.close()
    return summary


def main() -> None:
    results: dict[str, dict[str, float]] = {}

    momentum_bars = {s: load_bars(s, START, END, use_live=False) for s in MOMENTUM_SYMBOLS}
    momentum_engine = BacktestEngine(
        symbols=MOMENTUM_SYMBOLS,
        start=START,
        end=END,
        initial_capital=100_000.0,
        strategy_class=MomentumCrossoverStrategy,
        strategy_kwargs={"symbols": MOMENTUM_SYMBOLS},
        bars=momentum_bars,
    )
    results["Momentum"] = _run_strategy("Momentum", momentum_engine)

    symbol_a, symbol_b = PAIRS[0]
    pair_a, pair_b = generate_cointegrated_pair(symbol_a, symbol_b, START, END)
    pairs_engine = BacktestEngine(
        symbols=[symbol_a, symbol_b],
        start=START,
        end=END,
        initial_capital=100_000.0,
        strategy_class=PairsTradingStrategy,
        strategy_kwargs={"pair": (symbol_a, symbol_b)},
        bars={symbol_a: pair_a, symbol_b: pair_b},
    )
    results["Pairs Trading"] = _run_strategy("Pairs Trading", pairs_engine)

    mean_rev_bars = {s: load_bars(s, START, END, use_live=False) for s in MEAN_REV_SYMBOLS}
    mean_rev_engine = BacktestEngine(
        symbols=MEAN_REV_SYMBOLS,
        start=START,
        end=END,
        initial_capital=100_000.0,
        strategy_class=MeanReversionStrategy,
        strategy_kwargs={"symbols": MEAN_REV_SYMBOLS},
        bars=mean_rev_bars,
    )
    results["Mean Reversion"] = _run_strategy("Mean Reversion", mean_rev_engine)

    spy_bars = load_bars("SPY", START, END, use_live=False)
    df = pd.DataFrame(spy_bars).set_index("timestamp")

    returns = df["adj_close"].pct_change().dropna()
    equity = (1 + returns).cumprod() * 100_000.0
    curve = pd.DataFrame({"total": equity, "returns": returns})
    spy_summary = generate_summary(curve)
    spy_summary["hit_rate"] = 0.0
    spy_summary["ic"] = 0.0
    results["SPY"] = spy_summary

    wf_bars = {"SPY": spy_bars}
    wf = WalkForwardEngine(
        symbols=["SPY"],
        start=START,
        end=END,
        strategy_class=MomentumCrossoverStrategy,
        param_grid={"fast_window": [5, 10], "slow_window": [30, 50]},
        base_strategy_kwargs={"symbols": ["SPY"]},
        bars=wf_bars,
    )
    oos, is_curve = wf.run()
    if not oos.empty and not is_curve.empty:
        plt.figure(figsize=(10, 5))
        is_curve["total"].plot(label="in-sample", alpha=0.8)
        oos["total"].plot(label="out-of-sample", alpha=0.8)
        plt.legend()
        plt.title("Walk-forward momentum: in-sample vs out-of-sample")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "walk_forward.png")
        plt.close()

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
