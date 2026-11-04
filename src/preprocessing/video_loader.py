"""Đọc và kiểm tra video bằng OpenCV.

- Kiểm tra định dạng (MP4/AVI/MOV), file mở được, có FPS và số frame hợp lệ.
- Chuẩn hóa FPS bằng cách lấy mẫu theo dấu thời gian (không phụ thuộc FPS gốc).
- Resize giữ tỉ lệ, loại frame lỗi (None, đen, đồng màu).
- Quét dataset theo cấu trúc dataset/<split>/<nhãn>/<video>.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_EXTS = (".mp4", ".avi", ".mov")


class VideoError(Exception):
    """Video không hợp lệ hoặc không đọc được."""


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass
class VideoSample:
    path: Path
    label: str
    split: str

    @property
    def signer(self) -> str:
        """Tên file dạng <nhan>__<nguoi>__<so>.mp4 → trả về <nguoi>; ngược lại 'unknown'."""
        parts = self.path.stem.split("__")
        return parts[1] if len(parts) >= 3 else "unknown"


def probe_video(path: str | Path, extensions=DEFAULT_EXTS) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Không tồn tại: {path}")
    if path.suffix.lower() not in {e.lower() for e in extensions}:
        raise VideoError(f"Định dạng không hỗ trợ ({path.suffix}): {path.name}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"OpenCV không mở được: {path.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise VideoError(f"Kích thước frame không hợp lệ: {path.name}")
    if fps <= 0 or fps > 240:
        log.warning("FPS bất thường (%.2f) ở %s — giả định 30.", fps, path.name)
        fps = 30.0
    return VideoInfo(path, fps, count, w, h)


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    m = max(h, w)
    if max_side <= 0 or m <= max_side:
        return frame
    s = max_side / m
    return cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def is_valid_frame(frame: np.ndarray | None, min_brightness: float = 8.0, min_std: float = 2.0) -> bool:
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return False
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) >= min_brightness and float(gray.std()) >= min_std


def iter_frames(path: str | Path, target_fps: float | None = 15, max_side: int = 640,
                min_brightness: float = 8.0, min_std: float = 2.0,
                extensions=DEFAULT_EXTS) -> Iterator[tuple[float, np.ndarray]]:
    """Sinh (timestamp_giây, frame_BGR) đã chuẩn hóa FPS và kích thước.

    Lấy mẫu theo thời gian: frame được giữ khi timestamp vượt mốc kế tiếp 1/target_fps.
    """
    info = probe_video(path, extensions)
    cap = cv2.VideoCapture(str(info.path))
    if not cap.isOpened():
        raise VideoError(f"OpenCV không mở được: {info.path.name}")
    step = 1.0 / target_fps if target_fps and target_fps < info.fps else 0.0
    next_t = 0.0
    idx = 0
    dropped = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / info.fps
            idx += 1
            if step and t + 1e-6 < next_t:
                continue
            next_t = (next_t + step) if step else t
            if not is_valid_frame(frame, min_brightness, min_std):
                dropped += 1
                continue
            yield t, resize_max_side(frame, max_side)
    finally:
        cap.release()
        if dropped:
            log.debug("%s: loại %d frame lỗi", info.path.name, dropped)


def scan_dataset(dataset_dir: str | Path, splits=("train", "val", "test"),
                 extensions=DEFAULT_EXTS) -> list[VideoSample]:
    """Quét dataset/<split>/<nhãn>/*.{mp4,avi,mov}."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {root}")
    exts = {e.lower() for e in extensions}
    samples: list[VideoSample] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            log.warning("Thiếu thư mục split: %s", split_dir)
            continue
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(label_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in exts:
                    samples.append(VideoSample(f, label_dir.name, split))
    return samples


def validate_dataset(dataset_dir: str | Path, extensions=DEFAULT_EXTS, min_frames: int = 8,
                     min_per_class: int = 5) -> dict:
    """Kiểm tra toàn bộ dataset, trả về báo cáo (không sửa file)."""
    samples = scan_dataset(dataset_dir, extensions=extensions)
    report = {"total": len(samples), "bad": [], "counts": {}, "signers": {}, "warnings": []}
    for s in samples:
        try:
            info = probe_video(s.path, extensions)
            if info.frame_count and info.frame_count < min_frames:
                raise VideoError(f"quá ngắn ({info.frame_count} frame)")
        except VideoError as exc:
            report["bad"].append((str(s.path), str(exc)))
            continue
        report["counts"].setdefault(s.split, {}).setdefault(s.label, 0)
        report["counts"][s.split][s.label] += 1
        report["signers"].setdefault(s.split, set()).add(s.signer)
    train_labels = set(report["counts"].get("train", {}))
    for split in ("val", "test"):
        labels = set(report["counts"].get(split, {}))
        if missing := train_labels - labels:
            report["warnings"].append(f"{split} thiếu nhãn: {sorted(missing)}")
        if extra := labels - train_labels:
            report["warnings"].append(f"{split} có nhãn không có trong train: {sorted(extra)}")
    for label, n in report["counts"].get("train", {}).items():
        if n < min_per_class:
            report["warnings"].append(f"Nhãn '{label}' chỉ có {n} video train (<{min_per_class}).")
    tr, te = report["signers"].get("train", set()), report["signers"].get("test", set())
    if overlap := (tr & te) - {"unknown"}:
        report["warnings"].append(f"Người ký hiệu xuất hiện ở cả train và test: {sorted(overlap)} "
                                  "→ kết quả test có thể bị thổi phồng.")
    report["signers"] = {k: sorted(v) for k, v in report["signers"].items()}
    return report
