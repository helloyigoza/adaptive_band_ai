from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from .config import TrainConfig
from .data import download_ohlcv, normalize_symbol
from .inference import run_indicator_from_ohlcv
from .plotting import plot_bands
from .signals import last_signal_text
from .train import train_model
from .utils import ensure_dir, install_uvloop_if_available, pct, select_device


def _add_shared_opts(parser: argparse.ArgumentParser, with_positional: bool) -> None:
    if with_positional:
        parser.add_argument("symbol")
        parser.add_argument("interval")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--seq-min", type=int, default=48)
    parser.add_argument("--seq-max", type=int, default=144)
    parser.add_argument("--eval-seq-len", type=int, default=96)
    parser.add_argument("--max-hold-bars", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--torch-threads", type=int, default=0)


def _add_train_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.78)
    parser.add_argument("--resume", default="auto", help="auto|none|<path>. auto: varsa last.pt'den devam et")
    parser.add_argument("--fresh", action="store_true", help="Eski checkpoint olsa bile sıfırdan başla (--resume none ile aynı)")


def _add_indicator_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Varsayılan: models/<symbol>_<interval>/cpu_model.pt veya best.pt")
    parser.add_argument("--bars", type=int, default=150)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adaptive Learning Band AI")

    # Legacy Turkish quick form: python3 main.py --egitim btc 1h [opts...]
    p.add_argument("--egitim", "--eğitim", nargs=2, metavar=("SYMBOL", "TIMEFRAME"), help="Eğitim modu kısa kullanım")
    p.add_argument("--indikator", "--indikatör", nargs=2, metavar=("SYMBOL", "TIMEFRAME"), help="İndikatör/grafik modu kısa kullanım")
    # Quick mode'un da train/indicator opsiyonlarını üst seviyede tanımasını sağla.
    _add_shared_opts(p, with_positional=False)
    _add_train_opts(p)
    _add_indicator_opts(p)

    sub = p.add_subparsers(dest="command")

    tr = sub.add_parser("train", help="Model eğit")
    _add_shared_opts(tr, with_positional=True)
    _add_train_opts(tr)

    ind = sub.add_parser("indicator", help="Kayıtlı modelle grafik üret")
    _add_shared_opts(ind, with_positional=True)
    _add_indicator_opts(ind)

    return p


def config_from_args(args: argparse.Namespace, symbol: Optional[str] = None, interval: Optional[str] = None) -> TrainConfig:
    return TrainConfig(
        symbol=normalize_symbol(symbol or args.symbol),
        interval=interval or args.interval,
        start=getattr(args, "start", "2022-01-01"),
        end=getattr(args, "end", None),
        seq_min=getattr(args, "seq_min", 48),
        seq_max=getattr(args, "seq_max", 144),
        eval_seq_len=getattr(args, "eval_seq_len", 96),
        max_hold_bars=getattr(args, "max_hold_bars", 50),
        epochs=getattr(args, "epochs", 25),
        batch_size=getattr(args, "batch_size", 128),
        lr=getattr(args, "lr", 2e-4),
        hidden_size=getattr(args, "hidden", 96),
        num_layers=getattr(args, "layers", 2),
        dropout=getattr(args, "dropout", 0.12),
        train_ratio=getattr(args, "train_ratio", 0.78),
        num_workers=getattr(args, "num_workers", 4),
        torch_threads=getattr(args, "torch_threads", 0),
        device=getattr(args, "device", "auto"),
    )


def resolve_model_path(cfg: TrainConfig, explicit: Optional[str] = None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {path}")
        return path
    cpu = cfg.model_dir / "cpu_model.pt"
    best = cfg.model_dir / "best.pt"
    last = cfg.model_dir / "last.pt"
    for path in [cpu, best, last]:
        if path.exists():
            return path
    raise FileNotFoundError(f"Model bulunamadı. Önce eğitim çalıştır: python3 main.py --egitim {cfg.symbol} {cfg.interval}")


def run_train(args: argparse.Namespace, quick_pair: Optional[list[str]] = None) -> None:
    if quick_pair:
        cfg = config_from_args(args, quick_pair[0], quick_pair[1])
    else:
        cfg = config_from_args(args)
    print(f"Eğitim başlıyor: {cfg.symbol} {cfg.interval} | start={cfg.start} | device={select_device(cfg.device)}")
    resume = getattr(args, "resume", "auto")
    if getattr(args, "fresh", False):
        resume = "none"
    result = train_model(cfg, resume=resume)
    m = result["metrics"]
    print("\nBitti.")
    print(f"Model klasörü : {result['model_dir']}")
    print(f"CPU model     : {result['cpu_model']}")
    print(f"Grafik        : {result['chart']}")
    print(f"Sinyal sayısı : {m.get('signals', 0)} | win={pct(float(m.get('win_rate', 0)))} | avg={pct(float(m.get('avg_return', 0)))}")


def run_indicator(args: argparse.Namespace, quick_pair: Optional[list[str]] = None) -> None:
    if quick_pair:
        cfg = config_from_args(args, quick_pair[0], quick_pair[1])
    else:
        cfg = config_from_args(args)
    model_path = resolve_model_path(cfg, getattr(args, "model", None))
    print(f"İndikatör çiziliyor: {cfg.symbol} {cfg.interval} | model={model_path}")
    raw = download_ohlcv(cfg.symbol, cfg.interval, cfg.start, cfg.end)
    device = "cpu" if getattr(args, "device", "auto") == "auto" else getattr(args, "device", "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"
    pred, metrics, bundle = run_indicator_from_ohlcv(
        model_path,
        raw,
        bars=getattr(args, "bars", 250),
        seq_len=getattr(args, "eval_seq_len", None),
        device=device,
    )
    out_dir = ensure_dir(bundle.config.output_dir)
    chart = plot_bands(pred, out_dir, f"{bundle.config.symbol} {bundle.config.interval} Adaptive Band - Indicator", metrics)
    print("\nBitti.")
    print(f"Grafik        : {chart}")
    print(f"Son sinyal    : {last_signal_text(pred)}")
    print(f"Sinyal sayısı : {metrics.get('signals', 0)} | win={pct(float(metrics.get('win_rate', 0)))} | avg={pct(float(metrics.get('avg_return', 0)))}")


def main(argv: Optional[list[str]] = None) -> None:
    install_uvloop_if_available()
    parser = build_parser()
    args = parser.parse_args(argv)

    # For quick mode, argparse still has defaults only; create missing attrs used by config.
    if args.egitim:
        run_train(args, args.egitim)
        return
    if args.indikator:
        run_indicator(args, args.indikator)
        return

    if args.command == "train":
        run_train(args)
    elif args.command == "indicator":
        run_indicator(args)
    else:
        parser.print_help()
        sys.exit(2)
