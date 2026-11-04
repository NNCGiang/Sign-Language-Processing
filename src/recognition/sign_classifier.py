"""Nạp checkpoint và phân loại chuỗi đặc trưng → (nhãn, độ tin cậy, top-k)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.detection.keypoint_extractor import feature_signature
from .feature_sequence import prepare_sequence
from .lstm_model import SignRecognitionModel

log = logging.getLogger(__name__)

CHECKPOINT_NAME = "best_model.pt"


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]
    accepted: bool  # độ tin cậy ≥ ngưỡng và không phải nhãn nền


def save_checkpoint(path: Path, model: SignRecognitionModel, labels: list[str], cfg, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.config,
        "labels": labels,
        "sequence": {"length": cfg.sequence.length, "use_velocity": cfg.sequence.use_velocity},
        "feature_signature": feature_signature(cfg),
        "extra": extra or {},
    }, path)


class SignClassifier:
    def __init__(self, checkpoint: str | Path, device: str = "cpu", confidence_threshold: float = 0.7,
                 background_labels=("_nen",), cfg=None):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Chưa có checkpoint: {checkpoint}. Hãy huấn luyện trước: python train.py train")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.labels: list[str] = list(ckpt["labels"])
        self.seq_len = int(ckpt["sequence"]["length"])
        self.use_velocity = bool(ckpt["sequence"]["use_velocity"])
        self.signature = ckpt.get("feature_signature", {})
        if cfg is not None:
            cur = feature_signature(cfg)
            for key in ("feature_dim", "hands_backend", "face_enabled"):
                if key in self.signature and self.signature[key] != cur[key]:
                    raise ValueError(
                        f"Cấu hình đặc trưng hiện tại ({key}={cur[key]}) khác lúc huấn luyện "
                        f"({key}={self.signature[key]}). Sửa configs/config.yaml cho khớp hoặc huấn luyện lại.")
        self.model = SignRecognitionModel(**ckpt["model_config"]).to(device).eval()
        self.model.load_state_dict(ckpt["model_state"])
        self.device = device
        self.threshold = confidence_threshold
        self.background = set(background_labels or [])
        self.extra = ckpt.get("extra", {})
        log.info("Đã nạp mô hình %s: %d nhãn, T=%d", checkpoint.name, len(self.labels), self.seq_len)

    @torch.no_grad()
    def predict_proba(self, raw_sequences: list[np.ndarray]) -> np.ndarray:
        x = np.stack([prepare_sequence(r, self.seq_len, self.use_velocity) for r in raw_sequences])
        logits = self.model(torch.from_numpy(x).to(self.device))
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, raw_sequence: np.ndarray, top_k: int = 3) -> Prediction:
        if raw_sequence is None or len(raw_sequence) < 2:
            raise ValueError("Chuỗi đặc trưng quá ngắn để phân loại")
        p = self.predict_proba([raw_sequence])[0]
        order = np.argsort(-p)[:max(1, top_k)]
        top = [(self.labels[i], float(p[i])) for i in order]
        label, conf = top[0]
        accepted = conf >= self.threshold and label not in self.background
        return Prediction(label, conf, top, accepted)
