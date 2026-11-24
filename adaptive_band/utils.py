from __future__ import annotations

import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_torch_threads(n: int) -> None:
    """n<=0 -> tum mantiksal cekirdekleri kullan."""
    if not n or n <= 0:
        n = os.cpu_count() or 1
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, n // 2))
    except RuntimeError:
        # interop threads sadece ilk cagrida ayarlanabilir; ikinci cagrida hata verir, sessiz gec.
        pass
    # MKL/OMP env hint'leri (alt kutuphanelere de iletmek icin)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))


def install_uvloop_if_available() -> None:
    if os.name != "posix":
        return
    try:
        import uvloop  # type: ignore
        uvloop.install()
    except Exception:
        pass


def backup_file(path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
    if not path.exists():
        return None
    backup_dir = backup_dir or path.parent / "backups"
    ensure_dir(backup_dir)
    dest = backup_dir / f"{path.stem}_{now_stamp()}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def timer() -> float:
    return time.perf_counter()
