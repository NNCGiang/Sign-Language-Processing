"""Công cụ quay video dataset bằng webcam.

    python tools/record_dataset.py --label xin_chao --signer nguoi01 --count 30

- Mỗi clip dài --duration giây, có đếm ngược. Người ký hiệu bắt đầu & kết thúc ở tư thế nghỉ (tay buông).
- File lưu: dataset_raw/<label>/<label>__<signer>__<số>.mp4  (ảnh KHÔNG lật gương)
- Phím: SPACE = quay clip tiếp theo, a = tự động quay liên tục, d = xóa clip vừa quay, q = thoát.
Sau khi quay đủ, chia dữ liệu theo người: python tools/split_dataset.py dataset_raw dataset --by-signer
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="tên nhãn không dấu, vd: xin_chao")
    ap.add_argument("--signer", required=True, help="mã người ký hiệu, vd: nguoi01")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--duration", type=float, default=2.5)
    ap.add_argument("--countdown", type=float, default=1.5)
    ap.add_argument("--out", default="dataset_raw")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30)
    args = ap.parse_args()

    if not args.label.replace("_", "").isalnum() or not args.label.isascii():
        sys.exit("Nhãn phải không dấu, chỉ gồm chữ/số/_ (vd: xin_chao)")
    if "__" in args.signer or "__" in args.label:
        sys.exit("Nhãn và mã người không được chứa '__'")
    out_dir = (ROOT / args.out / args.label) if not Path(args.out).is_absolute() else Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{args.label}__{args.signer}__*.mp4"))
    idx = (int(existing[-1].stem.split("__")[-1]) + 1) if existing else 1

    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        sys.exit(f"Không mở được webcam {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    recorded, auto, last_file = 0, False, None
    print(f"Quay '{args.label}' cho '{args.signer}' — SPACE: quay, a: tự động, d: xóa clip vừa quay, q: thoát")

    def preview(frame, msg, color=(0, 255, 0)):
        v = cv2.flip(frame, 1)
        cv2.putText(v, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(v, f"{args.label} | {args.signer} | {recorded}/{args.count}", (20, v.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Record dataset", v)
        return cv2.waitKey(1) & 0xFF

    try:
        while recorded < args.count:
            ok, frame = cap.read()
            if not ok:
                print("Mất tín hiệu webcam"); break
            k = preview(frame, "SPACE de quay" + (" (auto)" if auto else ""))
            if k == ord("q"):
                break
            if k == ord("a"):
                auto = not auto
            if k == ord("d") and last_file and last_file.exists():
                last_file.unlink(); recorded -= 1; idx -= 1; last_file = None
                print("Đã xóa clip vừa quay")
                continue
            if k != ord(" ") and not auto:
                continue
            t_end = time.time() + args.countdown
            while time.time() < t_end:
                ok, frame = cap.read()
                if ok:
                    preview(frame, f"Chuan bi... {t_end - time.time():.1f}", (0, 200, 255))
            h, w = frame.shape[:2]
            path = out_dir / f"{args.label}__{args.signer}__{idx:03d}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            t_end = time.time() + args.duration
            n = 0
            while time.time() < t_end:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)  # lưu ảnh gốc, không lật
                n += 1
                preview(frame, "REC", (0, 0, 255))
            writer.release()
            if n < 10:
                path.unlink(missing_ok=True)
                print("Clip quá ngắn, bỏ qua")
                continue
            last_file = path
            recorded += 1; idx += 1
            print(f"Đã lưu {path.name} ({n} frame)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Hoàn tất: {recorded} clip trong {out_dir}")


if __name__ == "__main__":
    main()
