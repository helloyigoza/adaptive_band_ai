from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def attach_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mid_slope"] = out["mid"].diff()
    out["trend_color"] = np.where(out["mid_slope"] >= 0, 1, -1)
    prev_color = out["trend_color"].shift(1)
    out["trend_buy"] = (prev_color < 0) & (out["trend_color"] > 0)
    out["trend_sell"] = (prev_color > 0) & (out["trend_color"] < 0)

    prev_close = out["close"].shift(1)
    prev_upper = out["upper"].shift(1)
    prev_lower = out["lower"].shift(1)
    out["rev_sell"] = (prev_close <= prev_upper) & (out["close"] > out["upper"])
    out["rev_buy"] = (prev_close >= prev_lower) & (out["close"] < out["lower"])

    out["buy_signal"] = out["trend_buy"] | out["rev_buy"]
    out["sell_signal"] = out["trend_sell"] | out["rev_sell"]
    return out


def evaluate_signal_holds(df: pd.DataFrame, max_hold_bars: int = 50) -> Dict[str, float]:
    """Evaluate buy/sell signals until the next opposite signal or max_hold_bars.

    This is not a full exchange backtest. It is a signal-quality diagnostic.
    """
    if df.empty or "buy_signal" not in df.columns:
        return {"signals": 0, "win_rate": 0.0, "avg_return": 0.0, "max_loss": 0.0}

    closes = df["close"].to_numpy(dtype=np.float64)
    buys = df["buy_signal"].to_numpy(dtype=bool)
    sells = df["sell_signal"].to_numpy(dtype=bool)
    returns: List[float] = []
    sides: List[int] = []

    n = len(df)
    for i in range(n - 1):
        side = 1 if buys[i] else -1 if sells[i] else 0
        if side == 0:
            continue
        max_j = min(n - 1, i + max_hold_bars)
        exit_j = max_j
        for j in range(i + 1, max_j + 1):
            if side == 1 and sells[j]:
                exit_j = j
                break
            if side == -1 and buys[j]:
                exit_j = j
                break
        raw = (closes[exit_j] / closes[i]) - 1.0
        returns.append(raw * side)
        sides.append(side)

    if not returns:
        return {"signals": 0, "win_rate": 0.0, "avg_return": 0.0, "max_loss": 0.0}
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "signals": int(len(arr)),
        "buy_signals": int(sum(1 for s in sides if s == 1)),
        "sell_signals": int(sum(1 for s in sides if s == -1)),
        "win_rate": float(np.mean(arr > 0)),
        "avg_return": float(np.mean(arr)),
        "median_return": float(np.median(arr)),
        "max_loss": float(np.min(arr)),
        "max_gain": float(np.max(arr)),
    }


def last_signal_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "VERI YOK"
    last = df.iloc[-1]
    if bool(last.get("buy_signal", False)):
        return "AL"
    if bool(last.get("sell_signal", False)):
        return "SAT"
    return "SINYAL YOK"
