"""Advanced causal features: Ehlers DSP, efficient OHLC volatility,
microstructure proxies and complexity measures.

All transforms are strictly causal: every value at index ``i`` depends only
on samples at indices ``<= i``. They are intentionally written in plain
NumPy/Pandas so the project keeps a single dependency surface.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPS = 1e-12


# ---------------------------------------------------------------------------
# Ehlers DSP family
# ---------------------------------------------------------------------------

def ehlers_super_smoother(s: pd.Series, period: int = 10) -> pd.Series:
    """John Ehlers 2-pole Super Smoother (Butterworth-style).

    Output has substantially less lag than an EMA of equal period.
    Reference: Ehlers, "Cycle Analytics for Traders".
    """
    x = s.to_numpy(dtype=np.float64)
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    a = np.exp(-1.414 * np.pi / period)
    b = 2.0 * a * np.cos(1.414 * np.pi / period)
    c2 = b
    c3 = -a * a
    c1 = 1.0 - c2 - c3
    for i in range(n):
        x0 = x[i] if np.isfinite(x[i]) else 0.0
        x1 = x[i - 1] if i >= 1 and np.isfinite(x[i - 1]) else x0
        y1 = out[i - 1] if i >= 1 else x0
        y2 = out[i - 2] if i >= 2 else x0
        out[i] = c1 * 0.5 * (x0 + x1) + c2 * y1 + c3 * y2
    return pd.Series(out, index=s.index)


def ehlers_high_pass(s: pd.Series, period: int = 48) -> pd.Series:
    """Ehlers 2-pole High-Pass filter. Removes trends >= ``period`` bars,
    leaving a detrended cycle signal."""
    x = s.to_numpy(dtype=np.float64)
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    alpha1 = (np.cos(0.707 * 2.0 * np.pi / period) + np.sin(0.707 * 2.0 * np.pi / period) - 1.0) / np.cos(
        0.707 * 2.0 * np.pi / period
    )
    k1 = (1.0 - alpha1 / 2.0) ** 2
    k2 = 2.0 * (1.0 - alpha1)
    k3 = (1.0 - alpha1) ** 2
    for i in range(n):
        x0 = x[i] if np.isfinite(x[i]) else 0.0
        x1 = x[i - 1] if i >= 1 and np.isfinite(x[i - 1]) else x0
        x2 = x[i - 2] if i >= 2 and np.isfinite(x[i - 2]) else x0
        y1 = out[i - 1] if i >= 1 else 0.0
        y2 = out[i - 2] if i >= 2 else 0.0
        out[i] = k1 * (x0 - 2.0 * x1 + x2) + k2 * y1 - k3 * y2
    return pd.Series(out, index=s.index)


def hilbert_homodyne_period(s: pd.Series) -> pd.Series:
    """Ehlers Homodyne Discriminator dominant-cycle period.

    Returns an instantaneous estimate (in bars) of the dominant cycle, clipped
    to [6, 50]. This is a genuinely causal "leading" feature: it changes
    immediately when the cycle structure changes.
    """
    x = s.to_numpy(dtype=np.float64)
    n = len(x)
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.zeros(n)
    period[0] = 20.0

    def g(arr, idx):
        return arr[idx] if 0 <= idx < n else 0.0

    for i in range(n):
        if i < 6:
            period[i] = 20.0
            continue
        smooth[i] = (4.0 * x[i] + 3.0 * x[i - 1] + 2.0 * x[i - 2] + x[i - 3]) / 10.0
        period_prev = period[i - 1] if period[i - 1] > 0 else 20.0
        c = 0.075 * period_prev + 0.54
        detrender[i] = (
            0.0962 * smooth[i] + 0.5769 * g(smooth, i - 2)
            - 0.5769 * g(smooth, i - 4) - 0.0962 * g(smooth, i - 6)
        ) * c
        q1[i] = (
            0.0962 * detrender[i] + 0.5769 * g(detrender, i - 2)
            - 0.5769 * g(detrender, i - 4) - 0.0962 * g(detrender, i - 6)
        ) * c
        i1[i] = g(detrender, i - 3)
        ji[i] = (
            0.0962 * i1[i] + 0.5769 * g(i1, i - 2)
            - 0.5769 * g(i1, i - 4) - 0.0962 * g(i1, i - 6)
        ) * c
        jq[i] = (
            0.0962 * q1[i] + 0.5769 * g(q1, i - 2)
            - 0.5769 * g(q1, i - 4) - 0.0962 * g(q1, i - 6)
        ) * c
        i2_raw = i1[i] - jq[i]
        q2_raw = q1[i] + ji[i]
        i2[i] = 0.2 * i2_raw + 0.8 * (i2[i - 1] if i >= 1 else 0.0)
        q2[i] = 0.2 * q2_raw + 0.8 * (q2[i - 1] if i >= 1 else 0.0)
        re_raw = i2[i] * (i2[i - 1] if i >= 1 else 0.0) + q2[i] * (q2[i - 1] if i >= 1 else 0.0)
        im_raw = i2[i] * (q2[i - 1] if i >= 1 else 0.0) - q2[i] * (i2[i - 1] if i >= 1 else 0.0)
        re[i] = 0.2 * re_raw + 0.8 * (re[i - 1] if i >= 1 else 0.0)
        im[i] = 0.2 * im_raw + 0.8 * (im[i - 1] if i >= 1 else 0.0)
        if im[i] != 0.0 and re[i] != 0.0:
            p = 2.0 * np.pi / np.arctan2(im[i], re[i])
        else:
            p = period_prev
        if p > 1.5 * period_prev:
            p = 1.5 * period_prev
        if p < 0.67 * period_prev:
            p = 0.67 * period_prev
        p = float(np.clip(p, 6.0, 50.0))
        period[i] = 0.2 * p + 0.8 * period_prev
    return pd.Series(period, index=s.index)


def hilbert_phase_components(s: pd.Series, period_series: pd.Series) -> tuple:
    """Return (sin(phase), cos(phase)) using a simple causal Hilbert pair.

    Uses ``i1`` (real) and ``q1`` (quadrature) approximations from the
    homodyne discriminator. To keep things simple we recompute them here on
    the smoothed price.
    """
    x = s.to_numpy(dtype=np.float64)
    n = len(x)
    smooth = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    sin_p = np.zeros(n)
    cos_p = np.zeros(n)
    for i in range(n):
        if i < 6:
            continue
        smooth[i] = (4.0 * x[i] + 3.0 * x[i - 1] + 2.0 * x[i - 2] + x[i - 3]) / 10.0
        c = 0.075 * float(period_series.iloc[i - 1]) + 0.54
        i1[i] = smooth[i - 3]
        q1[i] = (
            0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]
        ) * c
        phase = np.arctan2(q1[i], i1[i] if i1[i] != 0 else EPS)
        sin_p[i] = np.sin(phase)
        cos_p[i] = np.cos(phase)
    return (
        pd.Series(sin_p, index=s.index),
        pd.Series(cos_p, index=s.index),
    )


# ---------------------------------------------------------------------------
# Efficient OHLC volatility estimators
# ---------------------------------------------------------------------------

def garman_klass_vol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Garman-Klass OHLC volatility (per bar, then rolling-mean over ``period``)."""
    o = np.log(df["open"].clip(lower=EPS))
    h = np.log(df["high"].clip(lower=EPS))
    l = np.log(df["low"].clip(lower=EPS))
    c = np.log(df["close"].clip(lower=EPS))
    rs = 0.5 * (h - l) ** 2 - (2.0 * np.log(2.0) - 1.0) * (c - o) ** 2
    return np.sqrt(rs.rolling(period, min_periods=period).mean().clip(lower=0.0))


def parkinson_vol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    h = np.log(df["high"].clip(lower=EPS))
    l = np.log(df["low"].clip(lower=EPS))
    rs = (h - l) ** 2 / (4.0 * np.log(2.0))
    return np.sqrt(rs.rolling(period, min_periods=period).mean().clip(lower=0.0))


def rogers_satchell_vol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    o = np.log(df["open"].clip(lower=EPS))
    h = np.log(df["high"].clip(lower=EPS))
    l = np.log(df["low"].clip(lower=EPS))
    c = np.log(df["close"].clip(lower=EPS))
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    return np.sqrt(rs.rolling(period, min_periods=period).mean().clip(lower=0.0))


def bipower_variation_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """Ratio of bipower variation to realised variance.
    Values < 1 indicate jumps. Strictly causal.
    """
    r = np.log(close.clip(lower=EPS)).diff()
    rv = (r ** 2).rolling(period, min_periods=period).sum()
    bv = (np.pi / 2.0) * (r.abs() * r.abs().shift(1)).rolling(period, min_periods=period).sum()
    return (bv / (rv + EPS)).clip(0.0, 2.0)


# ---------------------------------------------------------------------------
# Microstructure proxies
# ---------------------------------------------------------------------------

def amihud_illiquidity(df: pd.DataFrame, period: int = 20) -> pd.Series:
    r = np.log(df["close"].clip(lower=EPS)).diff().abs()
    dollar_vol = (df["close"] * df["volume"].fillna(0.0)).replace(0, np.nan)
    illiq = r / (dollar_vol + EPS)
    return illiq.rolling(period, min_periods=period).mean()


def roll_spread(close: pd.Series, period: int = 20) -> pd.Series:
    """Roll (1984) effective spread estimator from serial covariance of price changes."""
    dp = close.diff()
    cov = dp.rolling(period, min_periods=period).cov(dp.shift(1))
    spread2 = (-cov).clip(lower=0.0)
    return 2.0 * np.sqrt(spread2)


def kyle_lambda_proxy(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """|return| / signed_volume regressed via rolling sum approximation."""
    r = np.log(df["close"].clip(lower=EPS)).diff().abs()
    sv = (np.sign(df["close"].diff()).fillna(0.0) * df["volume"].fillna(0.0)).abs()
    num = r.rolling(period, min_periods=period).sum()
    den = sv.rolling(period, min_periods=period).sum()
    return num / (den + EPS)


# ---------------------------------------------------------------------------
# Complexity / regime measures
# ---------------------------------------------------------------------------

def _perm_entropy_window(arr: np.ndarray, order: int = 3) -> float:
    n = arr.shape[0]
    if n < order:
        return np.nan
    # Build all order-length permutations via argsort patterns
    idx = np.arange(n - order + 1)[:, None] + np.arange(order)[None, :]
    sub = arr[idx]
    ranks = np.argsort(sub, axis=1)
    # Hash each permutation pattern to a single integer
    weights = (order ** np.arange(order))[None, :]
    keys = (ranks * weights).sum(axis=1)
    _, counts = np.unique(keys, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p + EPS))
    return float(h / np.log(math.factorial(order)))


def permutation_entropy(s: pd.Series, window: int = 30, order: int = 3) -> pd.Series:
    return s.rolling(window, min_periods=window).apply(
        lambda a: _perm_entropy_window(a, order), raw=True
    )


def _hurst_rs_window(arr: np.ndarray) -> float:
    n = arr.shape[0]
    if n < 16:
        return np.nan
    arr = arr - np.mean(arr)
    cs = np.cumsum(arr)
    r = cs.max() - cs.min()
    s = arr.std(ddof=0)
    if s <= 0 or r <= 0:
        return np.nan
    return float(np.log(r / s) / np.log(n))


def hurst_rs(close: pd.Series, window: int = 64) -> pd.Series:
    r = np.log(close.clip(lower=EPS)).diff().fillna(0.0)
    return r.rolling(window, min_periods=window).apply(_hurst_rs_window, raw=True)


def realized_skew_kurt(close: pd.Series, window: int = 30) -> tuple:
    r = np.log(close.clip(lower=EPS)).diff()
    skew = r.rolling(window, min_periods=window).skew()
    kurt = r.rolling(window, min_periods=window).kurt()
    return skew, kurt
