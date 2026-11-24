from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FeatureConfig:
    norm_lookback: int = 128
    atr_period: int = 14
    ema_fast: int = 8
    ret_horizons: tuple = (1, 3, 5, 10)
    eff_period: int = 10
    donchian_period: int = 20
    vol_z_period: int = 20
    rv_short: int = 5
    rv_long: int = 30
    # Advanced (Ehlers DSP / microstructure / complexity) layer
    enable_advanced: bool = True
    ss_period: int = 10  # Ehlers Super Smoother period
    hp_period: int = 48  # Ehlers High-Pass detrend period
    gk_period: int = 20  # Garman-Klass / Parkinson / Rogers-Satchell window
    bv_period: int = 20  # Bipower variation window
    amihud_period: int = 20
    roll_period: int = 20
    kyle_period: int = 20
    perm_window: int = 30
    perm_order: int = 3
    hurst_window: int = 64
    skew_window: int = 30


@dataclass
class TrainConfig:
    symbol: str
    interval: str
    start: str = "2022-01-01"
    end: Optional[str] = None
    seq_min: int = 48
    seq_max: int = 144
    eval_seq_len: int = 96
    max_hold_bars: int = 50
    epochs: int = 25
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 1e-4
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.12
    max_offset: float = 0.65
    min_offset: float = 0.005
    train_ratio: float = 0.78
    seed: int = 42
    num_workers: int = 0
    torch_threads: int = 0
    device: str = "auto"
    feature: FeatureConfig = field(default_factory=FeatureConfig)

    @property
    def safe_name(self) -> str:
        return safe_run_name(self.symbol, self.interval)

    @property
    def model_dir(self) -> Path:
        return Path("models") / self.safe_name

    @property
    def output_dir(self) -> Path:
        return Path("outputs") / self.safe_name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        feature = d.get("feature", {})
        if isinstance(feature, dict):
            allowed = {f.name for f in FeatureConfig.__dataclass_fields__.values()}
            feat = {k: v for k, v in feature.items() if k in allowed}
            if "ret_horizons" in feat and isinstance(feat["ret_horizons"], list):
                feat["ret_horizons"] = tuple(feat["ret_horizons"])
            d["feature"] = FeatureConfig(**feat)
        allowed_top = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in allowed_top}
        return cls(**d)


@dataclass
class RuntimeBundle:
    config: TrainConfig
    feature_columns: List[str]
    target_columns: List[str]
    input_size: int


def safe_run_name(symbol: str, interval: str) -> str:
    raw = f"{symbol}_{interval}".replace("/", "_").replace("-", "_").replace(".", "_")
    return "".join(ch for ch in raw.upper() if ch.isalnum() or ch == "_")
