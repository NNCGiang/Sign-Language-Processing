"""Nhận diện ngôn ngữ ký hiệu từ video có sẵn (dòng lệnh).

    python predict.py video.mp4                 # in ký hiệu + câu tiếng Việt
    python predict.py video.mp4 --speak         # và đọc to bằng giọng tiếng Việt
    python predict.py thu_muc_video/ --json out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.utils import load_config, resolve_path, select_device, setup_logging

log = logging.getLogger("predict")


def main() -> int:
    ap = argparse.ArgumentParser(description="Dịch video ngôn ngữ ký hiệu Việt Nam sang văn bản/giọng nói")
    ap.add_argument("input", help="file video (mp4/avi/mov) hoặc thư mục")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--speak", action="store_true", help="đọc kết quả bằng TTS tiếng Việt")
    ap.add_argument("--json", default=None, help="lưu kết quả ra file JSON")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.project.log_level)
    from src.preprocessing.video_loader import VideoError
    from src.realtime.webcam_recognition import RealtimeRecognizer, recognize_video_file

    inp = Path(args.input)
    exts = {e.lower() for e in cfg.video.extensions}
    videos = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.suffix.lower() in exts)
    if not videos:
        log.error("Không tìm thấy video: %s", inp)
        return 1
    try:
        rec = RealtimeRecognizer(cfg, Path(args.checkpoint) if args.checkpoint else None,
                                 select_device(cfg.yolo.device), enable_tts=args.speak)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    results = []
    try:
        for v in videos:
            try:
                r = recognize_video_file(cfg, None, v, speak=args.speak, recognizer=rec)
            except VideoError as exc:
                log.error("%s: %s", v.name, exc)
                continue
            results.append(r)
            signs = " ".join(f"[{s}]" for s in r["display_signs"]) or "(không nhận diện được ký hiệu đủ tin cậy)"
            print(f"\n=== {v.name} ===")
            print(f"Ký hiệu nhận diện: {signs}")
            print(f"Văn bản dịch     : {r['text'] or '—'}")
            if r["whole_video"]:
                w = r["whole_video"]
                tops = ", ".join(f"{l} {p:.2f}" for l, p in w["top_k"])
                print(f"Dự đoán cả video : {tops}" + ("" if w["accepted"] else "  (dưới ngưỡng tin cậy)"))
            print(f"Tốc độ xử lý     : {r['processing_fps']:.1f} frame/s ({r['frames']} frame)")
    finally:
        rec.close()
    if args.json:
        out = resolve_path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Đã lưu %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
