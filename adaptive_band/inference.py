from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import RuntimeBundle, TrainConfig
from .features import add_features
from .model import AdaptiveBandGRU
from .signals import attach_signals, evaluate_signal_holds


def load_bundle(model_path: Path, map_location: str = "cpu") -> Tuple[AdaptiveBandGRU, RuntimeBundle, Dict]:
    payload = torch.load(model_path, map_location=map_location)
    cfg = TrainConfig.from_dict(payload["config"])
    feature_cols = payload["feature_columns"]
    target_cols = payload.get("target_columns", ["target_mid_norm", "target_up_offset", "target_down_offset", "target_dir"])
    input_size = int(payload.get("input_size", len(feature_cols)))
    model = AdaptiveBandGRU(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_offset=cfg.max_offset,
        min_offset=cfg.min_offset,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    bundle = RuntimeBundle(config=cfg, feature_columns=feature_cols, target_columns=target_cols, input_size=input_size)
    return model, bundle, payload


def predict_dataframe(
    model: AdaptiveBandGRU,
    df_features: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    device: torch.device,
) -> pd.DataFrame:
    model = model.to(device)
    model.eval()
    d = df_features.copy()
    feats = d[feature_cols].to_numpy(dtype=np.float32)
    mids = np.full(len(d), np.nan, dtype=np.float64)
    uppers = np.full(len(d), np.nan, dtype=np.float64)
    lowers = np.full(len(d), np.nan, dtype=np.float64)
    dir_prob = np.full(len(d), np.nan, dtype=np.float64)

    with torch.inference_mode():
        for end in range(max(4, seq_len) - 1, len(d)):
            start = max(0, end - seq_len + 1)
            x = torch.from_numpy(feats[start : end + 1]).unsqueeze(0).to(device)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)
            pred = model(x, lengths)
            pmin = float(d.iloc[end]["price_min"])
            pscale = float(d.iloc[end]["price_scale"])
            mid_norm = float(pred["mid_norm"].cpu().item())
            upper_norm = float(pred["upper_norm"].cpu().item())
            lower_norm = float(pred["lower_norm"].cpu().item())
            mids[end] = pmin + mid_norm * pscale
            uppers[end] = pmin + upper_norm * pscale
            lowers[end] = pmin + lower_norm * pscale
            dir_prob[end] = float(torch.sigmoid(pred["direction_logit"]).cpu().item())

    d["mid"] = mids
    d["upper"] = uppers
    d["lower"] = lowers
    d["dir_prob"] = dir_prob
    d = d.dropna(subset=["mid", "upper", "lower"])
    return attach_signals(d)


def run_indicator_from_ohlcv(
    model_path: Path,
    ohlcv: pd.DataFrame,
    bars: int = 250,
    seq_len: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Dict[str, float], RuntimeBundle]:
    model, bundle, _ = load_bundle(model_path, map_location="cpu")
    cfg = bundle.config
    d, _ = add_features(ohlcv, cfg.feature)
    seq = int(seq_len or cfg.eval_seq_len)
    pred = predict_dataframe(model, d, bundle.feature_columns, seq, torch.device(device))
    metrics = evaluate_signal_holds(pred, cfg.max_hold_bars)
    if bars and len(pred) > bars:
        pred_view = pred.iloc[-bars:].copy()
        metrics = evaluate_signal_holds(pred_view, cfg.max_hold_bars)
        return pred_view, metrics, bundle
    return pred, metrics, bundle
