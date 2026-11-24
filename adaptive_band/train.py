from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import RuntimeBundle, TrainConfig
from .data import download_ohlcv, normalize_symbol
from .dataset import BandDataset, chronological_split, collate_band_batch
from .features import add_features, make_targets
from .inference import predict_dataframe
from .model import AdaptiveBandGRU, band_loss
from .plotting import plot_bands
from .signals import attach_signals, evaluate_signal_holds
from .utils import backup_file, ensure_dir, now_stamp, save_json, select_device, set_seed, set_torch_threads

TARGET_COLUMNS = ["target_mid_norm", "target_up_offset", "target_down_offset", "target_dir"]


def prepare_training_frame(cfg: TrainConfig) -> Tuple[pd.DataFrame, list[str]]:
    raw = download_ohlcv(cfg.symbol, cfg.interval, cfg.start, cfg.end)
    cfg.symbol = normalize_symbol(cfg.symbol)
    feats, feature_cols = add_features(raw, cfg.feature)
    full = make_targets(feats, cfg.max_hold_bars, cfg.max_offset)
    if len(full) < cfg.seq_max + cfg.max_hold_bars + 50:
        raise RuntimeError(
            f"Veri çok az: {len(full)} bar. Daha büyük timeframe, daha geniş tarih ya da daha küçük seq/max_hold dene."
        )
    return full, feature_cols


def save_checkpoint(
    path: Path,
    model: AdaptiveBandGRU,
    cfg: TrainConfig,
    feature_cols: list[str],
    epoch: int,
    metrics: Dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    ensure_dir(path.parent)
    backup_file(path)
    cpu_model = model.to("cpu")
    payload = {
        "version": "0.1.0",
        "created_at_utc": now_stamp(),
        "config": cfg.to_dict(),
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLUMNS,
        "input_size": len(feature_cols),
        "epoch": epoch,
        "metrics": metrics,
        "model_state": cpu_model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def _count_params(model: AdaptiveBandGRU) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_epoch(
    model: AdaptiveBandGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str = "",
    log_interval: int = 0,
) -> Tuple[Dict[str, float], float | None]:
    train = optimizer is not None
    model.train(train)
    total: Dict[str, float] = {"loss": 0.0, "mid": 0.0, "up": 0.0, "down": 0.0, "dir": 0.0, "width_reg": 0.0}
    count = 0
    grad_norm: float | None = None
    iterator = tqdm(loader, desc=desc or ("train" if train else "val"), leave=False, dynamic_ncols=True)
    for batch_idx, batch in enumerate(iterator):
        x = batch.x.to(device)
        lengths = batch.lengths.to(device)
        y = batch.y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x, lengths)
        loss, parts = band_loss(pred, y)
        if train:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if grad_norm is None or isinstance(gn, (int, float)):
                grad_norm = float(gn) if gn is not None else None
            optimizer.step()
        bs = x.shape[0]
        for k in total:
            total[k] += parts[k] * bs
        count += bs
        if train and log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total["loss"] / max(1, count)
            iterator.set_postfix(loss=f"{avg_loss:.5f}")
    return {k: v / max(1, count) for k, v in total.items()}, grad_norm


def _resolve_resume_path(cfg: TrainConfig, resume: str) -> Path | None:
    if not resume or resume.lower() == "none":
        return None
    if resume.lower() == "auto":
        for name in ("last.pt", "best.pt", "cpu_model.pt"):
            p = cfg.model_dir / name
            if p.exists():
                return p
        return None
    p = Path(resume)
    if not p.exists():
        raise FileNotFoundError(f"Resume checkpoint bulunamadı: {p}")
    return p


def train_model(cfg: TrainConfig, resume: str = "auto") -> Dict[str, object]:
    set_seed(cfg.seed)
    set_torch_threads(cfg.torch_threads)
    device = select_device(cfg.device)
    ensure_dir(cfg.model_dir)
    ensure_dir(cfg.output_dir)

    df, feature_cols = prepare_training_frame(cfg)
    train_df, test_df = chronological_split(df, cfg.train_ratio)

    train_ds = BandDataset(train_df, feature_cols, TARGET_COLUMNS, cfg.seq_min, cfg.seq_max, random_seq=True)
    val_ds = BandDataset(test_df, feature_cols, TARGET_COLUMNS, cfg.seq_min, min(cfg.seq_max, cfg.eval_seq_len), random_seq=False)

    loader_kwargs: Dict[str, object] = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_band_batch,
        pin_memory=device.type == "cuda",
    )
    if cfg.num_workers and cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    print(
        f"[cpu] torch_threads={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()} num_workers={cfg.num_workers}"
    )

    model = AdaptiveBandGRU(
        input_size=len(feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        max_offset=cfg.max_offset,
        min_offset=cfg.min_offset,
    ).to(device)
    total_params = _count_params(model)
    print(
        f"[info] device={device} model_params={total_params:,} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"features={len(feature_cols)} train_samples={len(train_df)} val_samples={len(test_df)}"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    best_val = float("inf")
    history: list[Dict[str, float]] = []
    start_epoch = 1

    resume_path = _resolve_resume_path(cfg, resume)
    if resume_path is not None:
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume_path, map_location=device)
        ck_features = ckpt.get("feature_columns", [])
        if ck_features and ck_features != feature_cols:
            print(f"[uyari] feature kolonlari farkli, resume atlandi: {resume_path}")
        else:
            try:
                model.load_state_dict(ckpt["model_state"])
                if "optimizer_state" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state"])
                    except Exception as exc:
                        print(f"[uyari] optimizer state yuklenemedi: {exc}")
                prev_epoch = int(ckpt.get("epoch", 0))
                start_epoch = prev_epoch + 1
                prev_metrics = ckpt.get("metrics", {}) or {}
                if "val_loss" in prev_metrics:
                    best_val = float(prev_metrics["val_loss"])
                print(f"[resume] {resume_path} -> epoch {prev_epoch} sonrasi devam (start={start_epoch})")
                model.to(device)
            except Exception as exc:
                print(f"[uyari] checkpoint yuklenemedi, sifirdan baslaniyor: {exc}")
                start_epoch = 1

    if start_epoch > cfg.epochs:
        print(f"[bilgi] kayitli epoch ({start_epoch - 1}) >= hedef ({cfg.epochs}). Daha fazla egitmek icin --epochs artir.")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.perf_counter()
        train_parts, grad_norm = _run_epoch(model, train_loader, optimizer, device, desc=f"train e{epoch}")
        with torch.no_grad():
            val_parts, _ = _run_epoch(model, val_loader, None, device, desc=f"val e{epoch}")
        elapsed = time.perf_counter() - t0
        scheduler.step(val_parts["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        row = {f"train_{k}": v for k, v in train_parts.items()}
        row.update({f"val_{k}": v for k, v in val_parts.items()})
        row["epoch"] = epoch
        history.append(row)
        gn_str = f"gn={grad_norm:.3f} " if grad_norm is not None else ""
        print(
            f"epoch {epoch:03d}/{cfg.epochs} | "
            f"time={elapsed:.1f}s lr={current_lr:.2e} | "
            f"train_loss={train_parts['loss']:.5f} val_loss={val_parts['loss']:.5f} | "
            f"val_mid={val_parts['mid']:.5f} val_up={val_parts['up']:.5f} val_down={val_parts['down']:.5f} val_dir={val_parts['dir']:.5f} | "
            f"{gn_str}"
            f"train_dir={train_parts['dir']:.5f}"
        )

        save_checkpoint(cfg.model_dir / "last.pt", model, cfg, feature_cols, epoch, row, optimizer)
        model.to(device)
        if val_parts["loss"] < best_val:
            best_val = val_parts["loss"]
            print(f"  [best] val_loss improved to {best_val:.5f} — saving best.pt & cpu_model.pt")
            save_checkpoint(cfg.model_dir / "best.pt", model, cfg, feature_cols, epoch, row, optimizer)
            save_checkpoint(cfg.model_dir / "cpu_model.pt", model, cfg, feature_cols, epoch, row, None)
            model.to(device)
        else:
            print(f"  [best] val_loss={val_parts['loss']:.5f} (best={best_val:.5f})")

    # Walk-forward style fixed-model inference over the test split.
    model.to(device)
    test_pred = predict_dataframe(model, test_df, feature_cols, cfg.eval_seq_len, device)
    test_pred = attach_signals(test_pred)
    metrics = evaluate_signal_holds(test_pred, cfg.max_hold_bars)
    metrics.update({"best_val_loss": best_val, "bars_total": len(df), "bars_train": len(train_df), "bars_test": len(test_df)})

    chart_path = plot_bands(test_pred, cfg.output_dir, f"{cfg.symbol} {cfg.interval} Adaptive Band - Walk Forward", metrics, bars=300)
    save_json(cfg.output_dir / f"history_{now_stamp()}.json", {"history": history, "metrics": metrics})

    return {
        "model_dir": str(cfg.model_dir),
        "best_model": str(cfg.model_dir / "best.pt"),
        "cpu_model": str(cfg.model_dir / "cpu_model.pt"),
        "chart": str(chart_path),
        "metrics": metrics,
    }
