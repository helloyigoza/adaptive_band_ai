from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class Batch:
    x: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor
    meta_index: torch.Tensor


class BandDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        seq_min: int,
        seq_max: int,
        random_seq: bool = True,
    ) -> None:
        if seq_min < 4 or seq_max < seq_min:
            raise ValueError("seq_min/seq_max geçersiz")
        self.df = df.reset_index(drop=False).copy()
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.seq_min = int(seq_min)
        self.seq_max = int(seq_max)
        self.random_seq = bool(random_seq)
        self.features = self.df[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        self.targets = self.df[self.target_cols].to_numpy(dtype=np.float32, copy=True)
        self.valid_end_indices = list(range(self.seq_min - 1, len(self.df)))

    def __len__(self) -> int:
        return len(self.valid_end_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        end = self.valid_end_indices[idx]
        max_len = min(self.seq_max, end + 1)
        if self.random_seq:
            seq_len = random.randint(self.seq_min, max_len) if max_len >= self.seq_min else max_len
        else:
            seq_len = max_len
        start = end - seq_len + 1
        x = torch.from_numpy(self.features[start : end + 1])
        y = torch.from_numpy(self.targets[end])
        return {"x": x, "y": y, "idx": torch.tensor(end, dtype=torch.long)}


def collate_band_batch(items: List[Dict[str, torch.Tensor]]) -> Batch:
    xs = [it["x"] for it in items]
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y = torch.stack([it["y"] for it in items], dim=0)
    meta_index = torch.stack([it["idx"] for it in items], dim=0)
    return Batch(x=padded, lengths=lengths, y=y, meta_index=meta_index)


def chronological_split(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.2 <= train_ratio <= 0.95:
        raise ValueError("train_ratio 0.2 ile 0.95 arasında olmalı")
    cut = int(len(df) * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
