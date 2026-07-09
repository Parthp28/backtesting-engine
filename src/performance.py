import numpy as np
import pandas as pd
from scipy import stats

RISK_FREE_RATE = 0.05
TRADING_DAYS = 252


def compute_sharpe_ratio(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns - RISK_FREE_RATE / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())


def compute_sortino_ratio(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    excess = returns.mean() - RISK_FREE_RATE / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess / downside.std())


def compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, int, int]:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    if drawdown.empty:
        return 0.0, 0, 0
    max_dd = float(drawdown.min())
    trough_pos = int(drawdown.values.argmin())
    peak_pos = int(equity_curve.iloc[: trough_pos + 1].values.argmax()) if trough_pos > 0 else 0
    return max_dd, peak_pos, trough_pos


def compute_calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    annualized = (1 + returns.mean()) ** TRADING_DAYS - 1
    max_dd, _, _ = compute_max_drawdown(equity_curve)
    if max_dd == 0:
        return 0.0
    return float(annualized / abs(max_dd))


def compute_information_coefficient(
    signals: pd.Series, forward_returns: pd.Series
) -> float:
    if len(signals) < 2 or len(forward_returns) < 2:
        return 0.0
    aligned = pd.concat([signals, forward_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    corr, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(corr) if not np.isnan(corr) else 0.0


def compute_hit_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    profitable = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return profitable / len(trades)


def compute_profit_factor(trades: list[dict]) -> float:
    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def generate_summary(equity_curve: pd.DataFrame) -> dict[str, float]:
    returns = equity_curve["returns"].dropna()
    eq = equity_curve["total"]
    max_dd, _, _ = compute_max_drawdown(eq)
    annualized = (1 + returns.mean()) ** TRADING_DAYS - 1 if len(returns) else 0.0

    return {
        "annualized_return": float(annualized),
        "sharpe_ratio": compute_sharpe_ratio(returns),
        "sortino_ratio": compute_sortino_ratio(returns),
        "max_drawdown": max_dd,
        "calmar_ratio": compute_calmar_ratio(returns, eq),
    }
