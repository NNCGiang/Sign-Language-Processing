"""Trích xuất frame từ video ra bộ nhớ hoặc ra thư mục ảnh (để kiểm tra/gán nhãn YOLO)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from .video_loader import DEFAULT_EXTS, iter_frames

log = logging.getLogger(__name__)


def extract_frames(video_path: str | Path, target_fps: float = 15, max_side: int = 640,
                   min_brightness: float = 8.0, min_std: float = 2.0,
                   extensions=DEFAULT_EXTS) -> tuple[list[float], list[np.ndarray]]:
    """Đọc toàn bộ frame hợp lệ của video (đã chuẩn hóa FPS & kích thước)."""
    times, frames = [], []
    for t, f in iter_frames(video_path, target_fps, max_side, min_brightness, min_std, extensions):
        times.append(t)
        frames.append(f)
    return times, frames


def save_frames(video_path: str | Path, out_dir: str | Path, target_fps: float = 5,
                max_side: int = 960, quality: int = 95) -> int:
    """Lưu frame ra JPG — dùng để tạo dữ liệu gán nhãn bàn tay cho fine-tune YOLO26."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    n = 0
    for i, (_, frame) in enumerate(iter_frames(video_path, target_fps, max_side)):
        cv2.imwrite(str(out_dir / f"{stem}_{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        n += 1
    log.info("Đã lưu %d frame từ %s vào %s", n, Path(video_path).name, out_dir)
    return n


def main() -> None:
    from src.utils import setup_logging
    ap = argparse.ArgumentParser(description="Trích frame từ video hoặc thư mục video")
    ap.add_argument("input", help="file video hoặc thư mục")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=5)
    ap.add_argument("--max-side", type=int, default=960)
    args = ap.parse_args()
    setup_logging()
    inp = Path(args.input)
    videos = [inp] if inp.is_file() else [p for p in inp.rglob("*") if p.suffix.lower() in DEFAULT_EXTS]
    total = sum(save_frames(v, args.out_dir, args.fps, args.max_side) for v in videos)
    log.info("Tổng cộng %d frame.", total)


if __name__ == "__main__":
    main()
