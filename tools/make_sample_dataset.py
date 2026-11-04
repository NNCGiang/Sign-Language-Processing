"""Tạo DỮ LIỆU MẪU TỔNG HỢP để kiểm tra pipeline chạy đúng từ đầu đến cuối.

⚠ ĐÂY KHÔNG PHẢI NGÔN NGỮ KÝ HIỆU. Các "nhãn" là các kiểu biến đổi hình học (xoay, lắc, co dãn)
áp lên một ảnh người thật có sẵn trong Ultralytics. Mục đích duy nhất: xác nhận các module
(đọc video → YOLO26 → đặc trưng → GRU/LSTM → đánh giá → dịch → TTS) kết nối và chạy được.
Độ chính xác trên bộ này KHÔNG nói lên điều gì về khả năng nhận diện ký hiệu thật.

    python tools/make_sample_dataset.py --out dataset_sample
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CLASSES = {
    "mau_xoay_trai": lambda p: (+22 * p, 1.0, 1.0),
    "mau_xoay_phai": lambda p: (-22 * p, 1.0, 1.0),
    "mau_lac": lambda p: (16 * math.sin(2 * math.pi * 1.5 * p), 1.0, 1.0),
    "mau_co_doc": lambda p: (0.0, 1.0, 1.0 - 0.3 * math.sin(math.pi * p)),
    "_nen": lambda p: (0.0, 1.0, 1.0),
}
SIGNERS = {"train": ["s1", "s2", "s3"], "val": ["s4"], "test": ["s5"]}


def find_person_image() -> np.ndarray:
    try:
        from ultralytics.utils import ASSETS
        for name in ("zidane.jpg", "bus.jpg"):
            p = Path(ASSETS) / name
            if p.is_file():
                img = cv2.imread(str(p))
                if img is not None:
                    return img
    except Exception:
        pass
    sys.exit("Không tìm thấy ảnh mẫu trong gói ultralytics (ultralytics/assets/zidane.jpg).")


def render(base: np.ndarray, angle: float, sx: float, sy: float, tx: float, ty: float,
           size=(480, 360)) -> np.ndarray:
    h, w = base.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    S = np.array([[sx, 0, (1 - sx) * c[0]], [0, sy, (1 - sy) * c[1]], [0, 0, 1]], np.float64)
    M = (np.vstack([M, [0, 0, 1]]) @ S)[:2]
    M[:, 2] += (tx * w, ty * h)
    out = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.resize(out, size, interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_sample")
    ap.add_argument("--per-signer", type=int, default=4, help="số video mỗi nhãn mỗi người")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    img = find_person_image()
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    n = 0
    for split, signers in SIGNERS.items():
        for si, signer in enumerate(signers):
            # mỗi "người" = một cách cắt khung + tông màu khác nhau
            zoom = 1.0 + 0.08 * (si % 3)
            tint = rng.uniform(0.85, 1.15, 3)
            for label, fn in CLASSES.items():
                d = out / split / label
                d.mkdir(parents=True, exist_ok=True)
                for k in range(args.per_signer):
                    dur = rng.uniform(1.6, 2.6)
                    T = int(dur * args.fps)
                    tx0, ty0 = rng.uniform(-0.05, 0.05, 2)
                    bright = rng.uniform(0.8, 1.2)
                    path = d / f"{label}__{signer}__{k:03d}.mp4"
                    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (480, 360))
                    for t in range(T):
                        p = t / max(T - 1, 1)
                        ang, sx, sy = fn(p)
                        jitter = rng.normal(0, 0.6)
                        fr = render(img, ang + jitter, sx * zoom, sy * zoom, tx0, ty0)
                        fr = np.clip(fr.astype(np.float32) * tint * bright + rng.normal(0, 4, fr.shape), 0, 255)
                        wr.write(fr.astype(np.uint8))
                    wr.release()
                    n += 1
    print(f"Đã tạo {n} video tổng hợp trong {out}")
    print("⚠ Dữ liệu tổng hợp chỉ để kiểm tra pipeline, KHÔNG phải ngôn ngữ ký hiệu.")


if __name__ == "__main__":
    main()
