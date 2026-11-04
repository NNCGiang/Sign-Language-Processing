"""Chuỗi đặc trưng theo thời gian: trích xuất & cache, nội suy độ dài, vận tốc, tăng cường dữ liệu."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.detection.keypoint_extractor import FEATURE_DIM, N_HAND, SLICES, KeypointExtractor
from src.preprocessing.video_loader import VideoError, VideoSample, iter_frames, scan_dataset

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- trích xuất
def extract_video_features(extractor: KeypointExtractor, video_path: Path, cfg) -> np.ndarray:
    """Video → mảng (T_gốc, FEATURE_DIM) theo FPS chuẩn hóa."""
    extractor.reset()
    feats = []
    dt = int(1000 / cfg.video.target_fps)
    for _, frame in iter_frames(video_path, cfg.video.target_fps, cfg.video.max_side,
                                cfg.video.min_brightness, cfg.video.min_std, tuple(cfg.video.extensions)):
        feats.append(extractor.process(frame, dt_ms=dt).features)
    if len(feats) < cfg.video.min_frames:
        raise VideoError(f"Chỉ có {len(feats)} frame hợp lệ (< {cfg.video.min_frames})")
    return np.stack(feats).astype(np.float32)


def feature_path(features_dir: Path, sample: VideoSample) -> Path:
    return features_dir / sample.split / sample.label / f"{sample.path.stem}.npy"


def extract_dataset(cfg, extractor: KeypointExtractor, dataset_dir: Path, features_dir: Path,
                    overwrite: bool = False) -> pd.DataFrame:
    """Trích đặc trưng cho mọi video trong dataset/, lưu .npy + manifest.csv."""
    samples = scan_dataset(dataset_dir, extensions=tuple(cfg.video.extensions))
    if not samples:
        raise RuntimeError(
            f"Không tìm thấy video nào trong {dataset_dir}. Cấu trúc cần có: "
            "dataset/train/<nhan>/*.mp4, dataset/val/<nhan>/..., dataset/test/<nhan>/...")
    rows = []
    t0 = time.perf_counter()
    n_frames_total = 0
    for i, s in enumerate(samples, 1):
        out = feature_path(features_dir, s)
        status, n, hand_ratio = "ok", 0, 0.0
        try:
            if out.is_file() and not overwrite:
                arr = np.load(out)
            else:
                arr = extract_video_features(extractor, s.path, cfg)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, arr)
                n_frames_total += len(arr)
            n = len(arr)
            hand_ratio = float((arr[:, SLICES["hand_flags"]].max(1) > 0).mean())
            if hand_ratio < 0.2 and cfg.hands.backend != "none":
                log.warning("%s: chỉ thấy tay ở %.0f%% frame — kiểm tra lại video.", s.path.name, 100 * hand_ratio)
        except (VideoError, ValueError) as exc:
            status = f"error: {exc}"
            log.error("Bỏ qua %s: %s", s.path, exc)
        rows.append({"split": s.split, "label": s.label, "signer": s.signer, "video": str(s.path),
                     "features": str(out), "frames": n, "hand_ratio": round(hand_ratio, 3), "status": status})
        if i % 10 == 0 or i == len(samples):
            log.info("Trích đặc trưng: %d/%d video", i, len(samples))
    dt = time.perf_counter() - t0
    if n_frames_total:
        log.info("Tốc độ trích đặc trưng: %.1f frame/s (%d frame mới, %.1fs)",
                 n_frames_total / dt, n_frames_total, dt)
    df = pd.DataFrame(rows)
    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_dir / "manifest.csv", index=False, encoding="utf-8")
    with (features_dir / "feature_config.json").open("w", encoding="utf-8") as f:
        from src.detection.keypoint_extractor import feature_signature
        json.dump(feature_signature(cfg), f, ensure_ascii=False, indent=2)
    return df


# ----------------------------------------------------------------------------- biến đổi chuỗi
def resample_sequence(seq: np.ndarray, length: int) -> np.ndarray:
    """Nội suy tuyến tính theo thời gian về `length` bước (chuẩn hóa tốc độ thực hiện)."""
    seq = np.asarray(seq, np.float32)
    n = len(seq)
    if n == 0:
        return np.zeros((length, seq.shape[-1] if seq.ndim == 2 else FEATURE_DIM), np.float32)
    if n == 1:
        return np.repeat(seq, length, axis=0)
    src = np.linspace(0, n - 1, length)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (src - lo)[:, None].astype(np.float32)
    out = seq[lo] * (1 - w) + seq[hi] * w
    # cờ có tay: không nội suy mà lấy gần nhất để giữ giá trị 0/1
    fl = SLICES["hand_flags"]
    if seq.shape[1] == FEATURE_DIM:
        nearest = np.where(w[:, 0] < 0.5, lo, hi)
        out[:, fl] = seq[nearest][:, fl]
    return out.astype(np.float32)


def add_velocity(seq: np.ndarray) -> np.ndarray:
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    return np.concatenate([seq, vel], axis=1).astype(np.float32)


def prepare_sequence(raw: np.ndarray, length: int, use_velocity: bool) -> np.ndarray:
    seq = resample_sequence(raw, length)
    return add_velocity(seq) if use_velocity else seq


def model_input_dim(use_velocity: bool) -> int:
    return FEATURE_DIM * (2 if use_velocity else 1)


def _xy_blocks():
    """Danh sách (slice, stride) của các khối tọa độ (x,y,...) chịu biến đổi hình học toàn cục."""
    b0 = SLICES["body"].start
    lh, rh = SLICES["left_hand"].start, SLICES["right_hand"].start
    return [(slice(b0, b0 + 39), 3), (slice(lh, lh + N_HAND * 3), 3),
            (slice(rh, rh + N_HAND * 3), 3), (SLICES["face"], 2)]


def augment_sequence(raw: np.ndarray, aug, rng: np.random.Generator) -> np.ndarray:
    """Tăng cường trên chuỗi đặc trưng thô (trước khi nội suy)."""
    seq = raw.copy()
    n = len(seq)
    lo, hi = aug.time_crop
    keep = max(4, int(round(n * rng.uniform(lo, hi))))
    if keep < n:
        start = rng.integers(0, n - keep + 1)
        seq = seq[start:start + keep]
    # biến đổi hình học: tỉ lệ, xoay, tịnh tiến — áp lên các tọa độ (đã chuẩn hóa theo vai)
    s = rng.uniform(*aug.scale)
    a = math.radians(rng.uniform(-aug.rotate_deg, aug.rotate_deg))
    ca, sa = math.cos(a), math.sin(a)
    tx, ty = rng.uniform(-aug.shift, aug.shift, 2)
    for sl, stride in _xy_blocks():
        block = seq[:, sl].reshape(len(seq), -1, stride)
        mask = np.abs(block[..., :2]).sum(-1, keepdims=True) > 0  # bỏ qua điểm thiếu (=0)
        x, y = block[..., 0].copy(), block[..., 1].copy()
        nx = s * (ca * x - sa * y) + tx
        ny = s * (sa * x + ca * y) + ty
        block[..., 0] = np.where(mask[..., 0], nx, 0)
        block[..., 1] = np.where(mask[..., 0], ny, 0)
        seq[:, sl] = block.reshape(len(seq), -1)
    # nhiễu trên các giá trị khác 0
    nz = seq != 0
    fl = SLICES["hand_flags"]
    noise = rng.normal(0, aug.noise_std, seq.shape).astype(np.float32)
    noise[:, fl] = 0
    seq = seq + noise * nz
    # mất tay ngẫu nhiên ở một số frame
    if aug.hand_dropout > 0:
        for i, name in enumerate(("left_hand", "right_hand")):
            drop = rng.random(len(seq)) < aug.hand_dropout
            seq[drop, SLICES[name]] = 0
            seq[drop, fl.start + i] = 0
    return seq.astype(np.float32)


# ----------------------------------------------------------------------------- Dataset PyTorch
class SignSequenceDataset:
    """Dataset PyTorch đọc .npy đã cache. Trả về (tensor (T, D), nhãn int)."""

    def __init__(self, manifest: pd.DataFrame, split: str, label_to_idx: dict[str, int],
                 length: int, use_velocity: bool, augment=None, seed: int = 0):
        df = manifest[(manifest["split"] == split) & (manifest["status"] == "ok")]
        df = df[df["label"].isin(label_to_idx)]
        self.items = [(Path(p), label_to_idx[l]) for p, l in zip(df["features"], df["label"])]
        self.length, self.use_velocity, self.augment = length, use_velocity, augment
        self.rng = np.random.default_rng(seed)
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, p: Path) -> np.ndarray:
        if p not in self._cache:
            self._cache[p] = np.load(p).astype(np.float32)
        return self._cache[p]

    def __getitem__(self, i: int):
        import torch
        p, y = self.items[i]
        raw = self._load(p)
        if self.augment is not None:
            raw = augment_sequence(raw, self.augment, self.rng)
        x = prepare_sequence(raw, self.length, self.use_velocity)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    @property
    def targets(self) -> list[int]:
        return [y for _, y in self.items]
