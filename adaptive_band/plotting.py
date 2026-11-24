from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .utils import ensure_dir, now_stamp, save_json


def _draw_candles(ax, d: pd.DataFrame) -> None:
    """Draw OHLC candlesticks on the given Axes."""
    if len(d) < 2:
        return
    x = mdates.date2num(d.index.to_pydatetime())
    # Width = 70% of average bar spacing.
    width = float(np.median(np.diff(x))) * 0.7 if len(x) > 1 else 0.01
    o = d["open"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    up_color = "#26a69a"   # green
    dn_color = "#ef5350"   # red

    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        color = up_color if ci >= oi else dn_color
        # Wick
        ax.vlines(xi, li, hi, color=color, linewidth=0.8, zorder=2)
        # Body
        body_low = min(oi, ci)
        body_height = max(abs(ci - oi), (hi - li) * 0.001)  # avoid zero-height
        rect = Rectangle(
            (xi - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
            zorder=3,
        )
        ax.add_patch(rect)


def plot_bands(
    df: pd.DataFrame,
    out_dir: Path,
    title: str,
    metrics: Optional[Dict[str, float]] = None,
    bars: Optional[int] = None,
) -> Path:
    ensure_dir(out_dir)
    d = df.copy()
    if bars and len(d) > bars:
        d = d.iloc[-bars:]

    fig, ax = plt.subplots(figsize=(16, 8))
    _draw_candles(ax, d)
    ax.plot(d.index, d["mid"], label="AI Mid", linewidth=1.4, color="#fbc02d", zorder=4)
    ax.plot(d.index, d["upper"], label="AI Upper", linewidth=1.0, color="#42a5f5", zorder=4)
    ax.plot(d.index, d["lower"], label="AI Lower", linewidth=1.0, color="#ab47bc", zorder=4)

    if "buy_signal" in d.columns:
        buy = d[d["buy_signal"]]
        sell = d[d["sell_signal"]]
        if not buy.empty:
            ax.scatter(buy.index, buy["close"], marker="^", s=80, label="AL")
        if not sell.empty:
            ax.scatter(sell.index, sell["close"], marker="v", s=80, label="SAT")
    if "rev_buy" in d.columns:
        rb = d[d["rev_buy"]]
        rs = d[d["rev_sell"]]
        if not rb.empty:
            ax.scatter(rb.index, rb["low"], marker="^", s=40, label="Reversal AL")
        if not rs.empty:
            ax.scatter(rs.index, rs["high"], marker="v", s=40, label="Reversal SAT")

    subtitle = ""
    if metrics:
        subtitle = f" | signals={metrics.get('signals', 0)} win={metrics.get('win_rate', 0.0)*100:.1f}% avg={metrics.get('avg_return', 0.0)*100:.2f}%"
    ax.set_title(title + subtitle)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    path = out_dir / f"chart_{now_stamp()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    if metrics is not None:
        save_json(path.with_suffix(".metrics.json"), metrics)
    return path
