from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class AdaptiveBandGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.12,
        max_offset: float = 0.65,
        min_offset: float = 0.005,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_offset = float(max_offset)
        self.min_offset = float(min_offset)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.body = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
        )
        self.mid_head = nn.Linear(hidden_size // 2, 1)
        self.up_head = nn.Linear(hidden_size // 2, 1)
        self.down_head = nn.Linear(hidden_size // 2, 1)
        self.dir_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        lengths_cpu = lengths.detach().cpu()
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        last = h[-1]
        z = self.body(self.norm(last))

        mid = torch.sigmoid(self.mid_head(z)).squeeze(-1)
        up = self.min_offset + self.max_offset * torch.sigmoid(self.up_head(z)).squeeze(-1)
        down = self.min_offset + self.max_offset * torch.sigmoid(self.down_head(z)).squeeze(-1)
        direction_logit = self.dir_head(z).squeeze(-1)
        upper = mid + up
        lower = mid - down
        return {
            "mid_norm": mid,
            "up_offset": up,
            "down_offset": down,
            "upper_norm": upper,
            "lower_norm": lower,
            "direction_logit": direction_logit,
        }


def band_loss(pred: dict[str, torch.Tensor], y: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    target_mid = y[:, 0]
    target_up = y[:, 1]
    target_down = y[:, 2]
    target_dir = y[:, 3]

    smooth = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss()
    mid_loss = smooth(pred["mid_norm"], target_mid)
    up_loss = smooth(pred["up_offset"], target_up)
    down_loss = smooth(pred["down_offset"], target_down)
    dir_loss = bce(pred["direction_logit"], target_dir)

    # Small regularizer prevents absurdly wide bands while still allowing learned asymmetry.
    width_reg = 0.02 * torch.mean(pred["up_offset"] + pred["down_offset"])
    loss = mid_loss + 0.75 * up_loss + 0.75 * down_loss + 0.35 * dir_loss + width_reg
    parts = {
        "loss": float(loss.detach().cpu()),
        "mid": float(mid_loss.detach().cpu()),
        "up": float(up_loss.detach().cpu()),
        "down": float(down_loss.detach().cpu()),
        "dir": float(dir_loss.detach().cpu()),
        "width_reg": float(width_reg.detach().cpu()),
    }
    return loss, parts
