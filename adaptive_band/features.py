from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig
from . import advanced_features as af

EPS = 1e-9


def _rolling_minmax_norm(s: pd.Series, lookback: int) -> pd.Series:
    lo = s.rolling(lookback, min_periods=max(5, lookback // 5)).min()
    hi = s.rolling(lookback, min_periods=max(5, lookback // 5)).max()
    return ((s - lo) / (hi - lo + EPS)).clip(-3.0, 3.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    parts = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    return parts.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_features(df: pd.DataFrame, cfg: FeatureConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Causal feature set focused on low-lag momentum, microstructure and regime.

    The previous version relied almost exclusively on heavily smoothed indicators
    (CCI, CMF, EMV, multi-stage Fisher), all of which lag. We replace them with
    instantaneous candle geometry, short-horizon log-returns, efficiency ratio,
    Donchian position, volume z-score and realised-vol ratios.
    """
    out = df.copy()

    # ---- Causal rolling price/volume scaling (geometry) ----
    price_lo = out[["open", "high", "low", "close"]].min(axis=1).rolling(
        cfg.norm_lookback, min_periods=max(10, cfg.norm_lookback // 4)
    ).min()
    price_hi = out[["open", "high", "low", "close"]].max(axis=1).rolling(
        cfg.norm_lookback, min_periods=max(10, cfg.norm_lookback // 4)
    ).max()
    price_scale = (price_hi - price_lo).replace(0, np.nan)

    out["price_min"] = price_lo
    out["price_max"] = price_hi
    out["price_scale"] = price_scale

    for col in ["open", "high", "low", "close"]:
        out[f"{col}_norm"] = ((out[col] - price_lo) / (price_scale + EPS)).clip(-2.0, 3.0)

    out["volume_norm"] = _rolling_minmax_norm(out["volume"].astype(float), cfg.norm_lookback).fillna(0.0)

    feature_cols: List[str] = ["open_norm", "high_norm", "low_norm", "close_norm", "volume_norm"]

    # ---- Candle microstructure (instantaneous, zero lag) ----
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    body = out["close"] - out["open"]
    out["body_ratio"] = (body / (rng + EPS)).clip(-1.0, 1.0).fillna(0.0)
    upper = out["high"] - out[["open", "close"]].max(axis=1)
    lower = out[["open", "close"]].min(axis=1) - out["low"]
    out["upper_wick"] = (upper / (rng + EPS)).clip(0.0, 1.0).fillna(0.0)
    out["lower_wick"] = (lower / (rng + EPS)).clip(0.0, 1.0).fillna(0.0)
    feature_cols += ["body_ratio", "upper_wick", "lower_wick"]

    # ---- ATR-normalised range expansion ----
    tr = true_range(out)
    atr_s = atr(out, cfg.atr_period)
    out["tr_over_atr"] = (tr / (atr_s + EPS)).clip(0.0, 8.0).fillna(0.0)
    feature_cols.append("tr_over_atr")

    # ---- Log-returns at multiple horizons (pure momentum) ----
    log_close = np.log(out["close"].clip(lower=EPS))
    for h in cfg.ret_horizons:
        col = f"ret_{h}"
        out[col] = log_close.diff(h)
        feature_cols.append(col)
    # Acceleration (second derivative of price) on the smallest horizon
    out["ret_accel"] = out[f"ret_{cfg.ret_horizons[0]}"].diff()
    feature_cols.append("ret_accel")

    # ---- Z-score of close vs short EMA, scaled by recent stdev ----
    ema_fast = out["close"].ewm(span=cfg.ema_fast, adjust=False, min_periods=cfg.ema_fast).mean()
    short_std = out["close"].rolling(cfg.ema_fast * 3, min_periods=cfg.ema_fast).std()
    out["close_z_ema"] = ((out["close"] - ema_fast) / (short_std + EPS)).clip(-5.0, 5.0)
    feature_cols.append("close_z_ema")

    # ---- Kaufman efficiency ratio (trend quality, low lag) ----
    n_eff = cfg.eff_period
    direction = (out["close"] - out["close"].shift(n_eff)).abs()
    volatility = out["close"].diff().abs().rolling(n_eff, min_periods=n_eff).sum()
    out["eff_ratio"] = (direction / (volatility + EPS)).clip(0.0, 1.0)
    feature_cols.append("eff_ratio")

    # ---- Donchian position (faster than EMA-based trend) ----
    n_d = cfg.donchian_period
    d_lo = out["low"].rolling(n_d, min_periods=n_d).min()
    d_hi = out["high"].rolling(n_d, min_periods=n_d).max()
    out["donchian_pos"] = ((out["close"] - d_lo) / (d_hi - d_lo + EPS)).clip(0.0, 1.0)
    feature_cols.append("donchian_pos")

    # ---- Volume z-score and signed (order-flow proxy) ----
    n_v = cfg.vol_z_period
    vol_mean = out["volume"].rolling(n_v, min_periods=n_v).mean()
    vol_std = out["volume"].rolling(n_v, min_periods=n_v).std()
    out["volume_z"] = ((out["volume"] - vol_mean) / (vol_std + EPS)).clip(-5.0, 5.0)
    sign_close = np.sign(out["close"].diff()).fillna(0.0)
    out["signed_volume_z"] = (sign_close * out["volume_z"]).clip(-5.0, 5.0)
    feature_cols += ["volume_z", "signed_volume_z"]

    # ---- Realised vol ratio (regime) ----
    r1 = log_close.diff()
    rv_s = r1.rolling(cfg.rv_short, min_periods=cfg.rv_short).std()
    rv_l = r1.rolling(cfg.rv_long, min_periods=cfg.rv_long).std()
    out["rv_ratio"] = (rv_s / (rv_l + EPS)).clip(0.0, 5.0)
    feature_cols.append("rv_ratio")

    # ---- Advanced layer: Ehlers DSP, OHLC vol, microstructure, complexity ----
    if getattr(cfg, "enable_advanced", False):
        # Ehlers Super Smoother + slope (low-lag trend)
        ss = af.ehlers_super_smoother(out["close"], cfg.ss_period)
        out["ss_dev"] = ((out["close"] - ss) / (ss.abs() + EPS)).clip(-0.5, 0.5)
        out["ss_slope"] = (ss.diff() / (ss.abs() + EPS)).clip(-0.1, 0.1)
        feature_cols += ["ss_dev", "ss_slope"]

        # High-pass detrended cycle component (z-scored)
        hp = af.ehlers_high_pass(out["close"], cfg.hp_period)
        hp_std = hp.rolling(cfg.hp_period, min_periods=cfg.hp_period).std()
        out["hp_z"] = (hp / (hp_std + EPS)).clip(-5.0, 5.0)
        feature_cols.append("hp_z")

        # Hilbert dominant cycle period (instantaneous, leading)
        period_s = af.hilbert_homodyne_period(out["close"])
        out["dc_period"] = ((period_s - 6.0) / (50.0 - 6.0)).clip(0.0, 1.0)
        feature_cols.append("dc_period")

        # Hilbert sin/cos phase computed on the detrended (high-pass) signal so
        # the long-term price trend does not dominate the phase estimate.
        sin_p, cos_p = af.hilbert_phase_components(hp, period_s)
        out["dc_sin"] = sin_p.clip(-1.0, 1.0)
        out["dc_cos"] = cos_p.clip(-1.0, 1.0)
        feature_cols += ["dc_sin", "dc_cos"]

        # Efficient OHLC volatility estimators, normalised by close-to-close
        # realised volatility (same dimensional units: per-bar log-return sigma).
        gk = af.garman_klass_vol(out, cfg.gk_period)
        pk = af.parkinson_vol(out, cfg.gk_period)
        rs_v = af.rogers_satchell_vol(out, cfg.gk_period)
        rv = log_close.diff().rolling(cfg.gk_period, min_periods=cfg.gk_period).std()
        out["gk_vol_z"] = (gk / (rv + EPS)).clip(0.0, 5.0)
        out["pk_vol_z"] = (pk / (rv + EPS)).clip(0.0, 5.0)
        out["rs_vol_z"] = (rs_v / (rv + EPS)).clip(0.0, 5.0)
        feature_cols += ["gk_vol_z", "pk_vol_z", "rs_vol_z"]

        # Bipower variation ratio (jump detector)
        out["bv_ratio"] = af.bipower_variation_ratio(out["close"], cfg.bv_period)
        feature_cols.append("bv_ratio")

        # Microstructure proxies (z-scored)
        amihud = af.amihud_illiquidity(out, cfg.amihud_period)
        a_med = amihud.rolling(cfg.norm_lookback, min_periods=cfg.amihud_period).median()
        out["amihud_z"] = (amihud / (a_med + EPS)).clip(0.0, 10.0).fillna(0.0)
        roll_s = af.roll_spread(out["close"], cfg.roll_period)
        r_med = roll_s.rolling(cfg.norm_lookback, min_periods=cfg.roll_period).median()
        out["roll_spread_z"] = (roll_s / (r_med + EPS)).clip(0.0, 10.0).fillna(0.0)
        kyle = af.kyle_lambda_proxy(out, cfg.kyle_period)
        k_med = kyle.rolling(cfg.norm_lookback, min_periods=cfg.kyle_period).median()
        out["kyle_z"] = (kyle / (k_med + EPS)).clip(0.0, 10.0).fillna(0.0)
        feature_cols += ["amihud_z", "roll_spread_z", "kyle_z"]

        # Complexity / regime
        out["perm_entropy"] = af.permutation_entropy(
            out["close"], cfg.perm_window, cfg.perm_order
        )
        out["hurst"] = af.hurst_rs(out["close"], cfg.hurst_window)
        skew, kurt = af.realized_skew_kurt(out["close"], cfg.skew_window)
        out["realized_skew"] = skew.clip(-5.0, 5.0)
        out["realized_kurt"] = kurt.clip(-5.0, 20.0)
        feature_cols += ["perm_entropy", "hurst", "realized_skew", "realized_kurt"]

    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_cols + ["price_min", "price_max", "price_scale", "close"])
    return out, feature_cols


def make_targets(df: pd.DataFrame, max_hold_bars: int = 50, max_offset: float = 0.65) -> pd.DataFrame:
    """Build labels with future used only as target feedback.

    target_mid_norm = future balance/median price normalized by current causal scale.
    target_up/down = asymmetric future excursion around that balance.
    """
    out = df.copy()
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    pmin = out["price_min"].to_numpy(dtype=np.float64)
    pscale = out["price_scale"].to_numpy(dtype=np.float64)

    n = len(out)
    t_mid = np.full(n, np.nan, dtype=np.float64)
    t_up = np.full(n, np.nan, dtype=np.float64)
    t_down = np.full(n, np.nan, dtype=np.float64)
    t_dir = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        j1 = i + 1
        j2 = min(n, i + 1 + max_hold_bars)
        if j2 <= j1 or not np.isfinite(pscale[i]) or pscale[i] <= EPS:
            continue
        fut_close = close[j1:j2]
        fut_high = high[j1:j2]
        fut_low = low[j1:j2]
        if len(fut_close) == 0:
            continue
        balance = np.nanmedian(fut_close)
        mid_norm = (balance - pmin[i]) / (pscale[i] + EPS)
        up = (np.nanmax(fut_high) - balance) / (pscale[i] + EPS)
        down = (balance - np.nanmin(fut_low)) / (pscale[i] + EPS)
        t_mid[i] = np.clip(mid_norm, -0.5, 1.5)
        t_up[i] = np.clip(max(up, 0.0), 0.0, max_offset)
        t_down[i] = np.clip(max(down, 0.0), 0.0, max_offset)
        t_dir[i] = 1.0 if fut_close[-1] > close[i] else 0.0

    out["target_mid_norm"] = t_mid
    out["target_up_offset"] = t_up
    out["target_down_offset"] = t_down
    out["target_dir"] = t_dir
    return out.dropna(subset=["target_mid_norm", "target_up_offset", "target_down_offset", "target_dir"])
