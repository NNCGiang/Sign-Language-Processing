"""Kiểm thử toàn bộ pipeline trên dữ liệu mẫu tổng hợp (không đụng tới dataset/ thật).

    python tools/smoke_test.py            # đầy đủ
    python tools/smoke_test.py --quick    # ít epoch hơn

Các bước: kiểm tra dịch & bộ giải mã (unit test) → tạo dữ liệu mẫu → check → extract (YOLO26)
→ train GRU → evaluate trên test → predict một video → TTS (nếu có mạng).
Kết quả đặt trong runs/smoke/.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, resolve_path, select_device, setup_logging  # noqa: E402

log = logging.getLogger("smoke")


def test_translator():
    from src.translation.vietnamese_translator import VietnameseTranslator
    tr = VietnameseTranslator(ROOT / "configs/labels.yaml", ROOT / "configs/grammar.yaml")
    cases = {
        ("xin_chao", "toi", "khoe"): "Xin chào! Tôi khỏe.",
        ("_nen", "cam_on", "cam_on", "_nen"): "Cảm ơn!",
        ("ban", "ten", "la_gi"): "Bạn tên là gì?",
        ("toi", "an", "khong"): "Tôi không ăn.",
        ("toi", "uong", "nuoc"): "Tôi uống nước.",
        ("ban", "o_dau"): "Bạn ở đâu?",
        ("tu_moi",): "Tu moi.",
    }
    for gl, exp in cases.items():
        got = tr.translate(list(gl)).text
        assert got == exp, f"{gl}: '{got}' != '{exp}'"
    assert tr.translate(["tu_moi"]).unknown == ["tu_moi"]
    log.info("✓ Bộ dịch: %d trường hợp đúng", len(cases))


def test_decoder():
    from src.realtime.webcam_recognition import ContinuousDecoder
    from src.recognition.sign_classifier import Prediction
    # mô phỏng: 2s ký hiệu A, 0.5s chuyển, 2s ký hiệu B, 2s nghỉ (không tay)
    script = [("A", True)] * 30 + [("?", True)] * 8 + [("B", True)] * 30 + [("_nen", False)] * 30

    state = {"i": 0}

    def fake_classify(seq):
        lab = script[min(state["i"], len(script) - 1)][0]
        if lab == "?":
            return Prediction("A", 0.4, [("A", 0.4)], False)
        return Prediction(lab, 0.95, [(lab, 0.95)], lab != "_nen")

    dec = ContinuousDecoder(fake_classify, window_sec=1.0, min_window_sec=0.4, stride_frames=2,
                            confirm_windows=2, cooldown_sec=0.5, idle_sec=1.0)
    events = []
    for i, (_, hand) in enumerate(script):
        state["i"] = i
        events += dec.update(i / 15, np.zeros(4, np.float32), hand)
    signs = [e.label for e in events if e.kind == "sign"]
    sents = [e.signs for e in events if e.kind == "sentence"]
    assert signs == ["A", "B"], signs
    assert sents == [["A", "B"]], sents
    log.info("✓ Bộ giải mã liên tục: phát hiện %s, kết thúc câu %s, không lặp nhãn", signs, sents)


def make_config(tmp_cfg: Path, try_mediapipe: bool) -> Path:
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    cfg["paths"].update({"dataset_dir": "dataset_sample", "features_dir": "runs/smoke/features",
                         "checkpoints_dir": "runs/smoke/checkpoints", "runs_dir": "runs/smoke"})
    cfg["inference"]["background_labels"] = ["_nen"]
    if try_mediapipe:
        from src.utils import download_file
        try:
            download_file(cfg["hands"]["mediapipe_url"], ROOT / cfg["paths"]["mediapipe_dir"] / cfg["hands"]["mediapipe_model"], 30)
            download_file(cfg["face"]["mediapipe_url"], ROOT / cfg["paths"]["mediapipe_dir"] / cfg["face"]["mediapipe_model"], 30)
        except RuntimeError as exc:
            log.warning("Không tải được mô hình MediaPipe (%s) → smoke test dùng hands.backend=none, face tắt.",
                        str(exc).splitlines()[0])
            try_mediapipe = False
    if not try_mediapipe:
        cfg["hands"]["backend"] = "none"
        cfg["face"]["enabled"] = False
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    tmp_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return tmp_cfg


def run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"Lệnh thất bại ({r.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-mediapipe", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    args = ap.parse_args()
    setup_logging("INFO")

    test_translator()
    test_decoder()
    cfg_path = make_config(ROOT / "runs/smoke/config_smoke.yaml", not args.no_mediapipe)
    if not (ROOT / "dataset_sample").is_dir():
        run(["tools/make_sample_dataset.py", "--out", "dataset_sample"])
    run(["train.py", "check", "--config", str(cfg_path)])
    run(["train.py", "extract", "--config", str(cfg_path)])
    run(["train.py", "train", "--config", str(cfg_path), "--epochs", "25" if args.quick else "60"])
    run(["-m", "src.evaluation.evaluate", "--config", str(cfg_path), "--out", "runs/smoke/evaluation",
         "--fps-videos", "3"])
    test_videos = sorted((ROOT / "dataset_sample/test/mau_lac").glob("*.mp4"))
    run(["predict.py", str(test_videos[0]), "--config", str(cfg_path), "--json", "runs/smoke/predict.json"])

    if not args.skip_tts:
        cfg = load_config(cfg_path)
        from src.speech.vietnamese_tts import TTSError, VietnameseTTS
        tts = VietnameseTTS.from_config(cfg)
        try:
            p = tts.synthesize("Xin chào! Tôi khỏe.")
            log.info("✓ TTS tạo được âm thanh: %s (%d byte)", p, p.stat().st_size)
        except TTSError as exc:
            log.warning("TTS không chạy được trong môi trường này: %s", str(exc).splitlines()[0])
    log.info("✓ Smoke test hoàn tất. Xem runs/smoke/evaluation/")


if __name__ == "__main__":
    main()
