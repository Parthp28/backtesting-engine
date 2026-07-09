"""Market data helpers and symbol configuration."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

MOMENTUM_SYMBOLS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
PAIRS = [("GLD", "SLV"), ("SPY", "IVV"), ("QQQ", "XLK")]
MEAN_REV_SYMBOLS = ["SPY", "GLD", "TLT", "EEM"]

START = "2010-01-01"
END = "2024-12-31"


def generate_synthetic_bars(
    symbol: str,
    start: str,
    end: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Build reproducible OHLCV bars when live downloads are unavailable."""
    rng = np.random.default_rng(seed or hash(symbol) % 10_000)
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    days = (end_dt - start_dt).days
    price = 100.0 + (hash(symbol) % 50)
    bars: list[dict[str, Any]] = []
    for i in range(days):
        drift = 0.0002 if symbol in {"SPY", "QQQ", "IWM", "IVV", "XLK"} else 0.0001
        shock = rng.normal(drift, 0.01)
        price = max(5.0, price * (1 + shock))
        spread = abs(rng.normal(0, 0.3))
        open_px = price
        close_px = price * (1 + rng.normal(0, 0.002))
        bars.append(
            {
                "timestamp": start_dt + timedelta(days=i),
                "open": float(open_px),
                "high": float(max(open_px, close_px) + spread),
                "low": float(min(open_px, close_px) - spread),
                "close": float(close_px),
                "volume": int(abs(rng.normal(2_000_000, 300_000))),
                "adj_close": float(close_px),
            }
        )
    return bars


def generate_cointegrated_pair(
    symbol_a: str,
    symbol_b: str,
    start: str,
    end: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a pair with stationary spread for pairs strategy demos."""
    rng = np.random.default_rng(42)
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    days = (end_dt - start_dt).days
    bars_b: list[dict[str, Any]] = []
    bars_a: list[dict[str, Any]] = []
    b_price = 50.0
    spread = 0.0
    for i in range(days):
        b_price = max(10.0, b_price * (1 + rng.normal(0.0002, 0.008)))
        spread = 0.9 * spread + rng.normal(0, 0.5)
        a_price = 2.0 * b_price + spread
        ts = start_dt + timedelta(days=i)
        for price, store in ((a_price, bars_a), (b_price, bars_b)):
            store.append(
                {
                    "timestamp": ts,
                    "open": float(price),
                    "high": float(price + 0.5),
                    "low": float(price - 0.5),
                    "close": float(price),
                    "volume": int(abs(rng.normal(2_000_000, 200_000))),
                    "adj_close": float(price),
                }
            )
    return bars_a, bars_b


def load_bars(symbol: str, start: str, end: str, use_live: bool = True) -> list[dict[str, Any]]:
    """Download bars from yfinance, falling back to synthetic data on failure."""
    if not use_live:
        return generate_synthetic_bars(symbol, start, end)
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError("empty download")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        bars: list[dict[str, Any]] = []
        for idx, row in df.iterrows():
            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            bars.append(
                {
                    "timestamp": ts,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "adj_close": float(row["Close"]),
                }
            )
        return bars
    except Exception:
        return generate_synthetic_bars(symbol, start, end)
