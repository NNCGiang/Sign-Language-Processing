"""Mô hình chuỗi thời gian: chiếu đặc trưng → BiGRU/BiLSTM → attention pooling → phân loại."""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Gộp chuỗi bằng trọng số học được — tập trung vào các frame mang thông tin của ký hiệu."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (B, T, D)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)  # (B, T)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1)       # (B, D)


class SignRecognitionModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, rnn_type: str = "gru", proj_dim: int = 256,
                 hidden_size: int = 128, num_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.config = dict(input_dim=input_dim, num_classes=num_classes, rnn_type=rnn_type,
                           proj_dim=proj_dim, hidden_size=hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=dropout)
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.LayerNorm(proj_dim),
                                  nn.GELU(), nn.Dropout(dropout))
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(rnn_type.lower())
        if rnn_cls is None:
            raise ValueError(f"rnn_type phải là 'gru' hoặc 'lstm', nhận '{rnn_type}'")
        self.rnn = rnn_cls(proj_dim, hidden_size, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.pool = AttentionPooling(out_dim)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, input_dim)
        h = self.proj(self.input_norm(x))
        h, _ = self.rnn(h)
        return self.head(self.pool(h))

    @classmethod
    def from_config(cls, cfg_model, input_dim: int, num_classes: int) -> "SignRecognitionModel":
        return cls(input_dim=input_dim, num_classes=num_classes, rnn_type=cfg_model.rnn_type,
                   proj_dim=cfg_model.proj_dim, hidden_size=cfg_model.hidden_size,
                   num_layers=cfg_model.num_layers, bidirectional=cfg_model.bidirectional,
                   dropout=cfg_model.dropout)
