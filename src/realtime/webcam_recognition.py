"""Nhận diện liên tục (webcam thời gian thực hoặc video có sẵn).

Cơ chế:
- Buffer đặc trưng theo thời gian (cửa sổ `window_sec`), phân loại lại mỗi `stride_frames` frame.
- Nhãn chỉ được chấp nhận khi top-1 giống nhau ở `confirm_windows` cửa sổ liên tiếp và
  độ tin cậy ≥ ngưỡng → chống nhiễu.
- Không nhận lặp cùng ký hiệu: cần một khoảng "ngắt" (cửa sổ khác nhãn/không tay) + `cooldown_sec`.
- Không thấy tay trong `idle_sec` → kết thúc câu → dịch → đọc cả câu.

Chạy webcam từ dòng lệnh:
    python -m src.realtime.webcam_recognition --camera 0
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.recognition.sign_classifier import Prediction

log = logging.getLogger(__name__)


@dataclass
class DecoderEvent:
    kind: str                 # "sign" | "sentence"
    label: str = ""
    confidence: float = 0.0
    text: str = ""
    signs: list[str] = field(default_factory=list)


class ContinuousDecoder:
    """Logic giải mã câu, tách khỏi phần camera để dễ kiểm thử."""

    def __init__(self, classify_fn, window_sec=2.0, min_window_sec=0.8, stride_frames=3, confirm_windows=3,
                 cooldown_sec=1.0, idle_sec=1.2, min_hand_ratio=0.3):
        self.classify_fn = classify_fn
        self.window_sec, self.min_window_sec = window_sec, min_window_sec
        self.stride, self.confirm = max(1, stride_frames), max(1, confirm_windows)
        self.cooldown, self.idle_sec, self.min_hand_ratio = cooldown_sec, idle_sec, min_hand_ratio
        self.reset()

    def reset(self) -> None:
        self.buf: deque[tuple[float, np.ndarray, bool]] = deque()
        self.n = 0
        self.cand, self.cand_count = None, 0
        self.last_emit_label, self.last_emit_t, self.had_break = None, -1e9, True
        self.last_hand_t = None
        self.signs: list[tuple[str, float]] = []
        self.current: Prediction | None = None

    def update(self, t: float, feat: np.ndarray, any_hand: bool) -> list[DecoderEvent]:
        events: list[DecoderEvent] = []
        self.buf.append((t, feat, any_hand))
        while self.buf and t - self.buf[0][0] > self.window_sec:
            self.buf.popleft()
        if any_hand:
            self.last_hand_t = t
        self.n += 1

        if self.n % self.stride == 0:
            dur = t - self.buf[0][0]
            hand_ratio = sum(h for _, _, h in self.buf) / len(self.buf)
            if dur >= self.min_window_sec and hand_ratio >= self.min_hand_ratio and len(self.buf) >= 4:
                pred = self.classify_fn(np.stack([f for _, f, _ in self.buf]))
                self.current = pred
                if pred.accepted:
                    if pred.label == self.cand:
                        self.cand_count += 1
                    else:
                        self.cand, self.cand_count = pred.label, 1
                        if pred.label != self.last_emit_label:
                            self.had_break = True
                    if self.cand_count >= self.confirm:
                        can_emit = (pred.label != self.last_emit_label or
                                    (self.had_break and t - self.last_emit_t >= self.cooldown))
                        if can_emit:
                            self.signs.append((pred.label, pred.confidence))
                            self.last_emit_label, self.last_emit_t = pred.label, t
                            self.had_break = False
                            self.cand, self.cand_count = None, 0
                            self.buf.clear()  # bắt đầu cửa sổ mới cho ký hiệu tiếp theo
                            events.append(DecoderEvent("sign", pred.label, pred.confidence))
                else:
                    self.cand, self.cand_count = None, 0
                    self.had_break = True
            elif hand_ratio < self.min_hand_ratio:
                # tay nghỉ → đây là khoảng ngắt thật giữa hai ký hiệu
                self.current = None
                self.cand, self.cand_count = None, 0
                self.had_break = True
            # else: cửa sổ chưa đủ dài (vừa xóa buffer) → chờ thêm, KHÔNG coi là ngắt

        if self.signs and self.last_hand_t is not None and t - self.last_hand_t >= self.idle_sec:
            events.append(self.end_sentence())
        return events

    def end_sentence(self) -> DecoderEvent:
        labels = [l for l, _ in self.signs]
        self.signs = []
        self.last_emit_label = None
        self.had_break = True
        self.buf.clear()
        return DecoderEvent("sentence", signs=labels)


class RealtimeRecognizer:
    """Ghép extractor + classifier + decoder + translator + TTS. Dùng chung cho GUI và CLI."""

    def __init__(self, cfg, checkpoint: Path | None = None, device: str | None = None, enable_tts: bool = True):
        from src.detection.keypoint_extractor import KeypointExtractor
        from src.recognition.sign_classifier import CHECKPOINT_NAME, SignClassifier
        from src.speech.vietnamese_tts import VietnameseTTS
        from src.translation.vietnamese_translator import VietnameseTranslator
        from src.utils import resolve_path, select_device
        self.cfg = cfg
        self.device = device or select_device(cfg.yolo.device)
        ckpt = checkpoint or resolve_path(cfg.paths.checkpoints_dir) / CHECKPOINT_NAME
        inf = cfg.inference
        self.classifier = SignClassifier(ckpt, self.device, inf.confidence_threshold, inf.background_labels, cfg)
        self.extractor = KeypointExtractor(cfg, self.device)
        self.translator = VietnameseTranslator.from_config(cfg)
        self.tts = VietnameseTTS.from_config(cfg) if enable_tts else None
        self.decoder = ContinuousDecoder(lambda seq: self.classifier.predict(seq, inf.top_k),
                                         inf.window_sec, inf.min_window_sec, inf.stride_frames,
                                         inf.confirm_windows, inf.cooldown_sec, inf.idle_sec)
        self.detect_interval = max(1, int(cfg.yolo.detect_interval))
        self.all_signs: list[str] = []     # mọi ký hiệu của phiên hiện tại (hiển thị)
        self.sentences: list[str] = []     # các câu đã dịch
        self.confidences: list[float] = []
        self.history_feats: list[np.ndarray] = []
        self._frame_i = 0
        self._fps = 0.0
        self._last_t = None
        self.tts_error: str | None = None

    # ------------------------------------------------------------------ xử lý
    def process_frame(self, frame_bgr: np.ndarray, t: float | None = None, dt_ms: int = 33,
                      draw: bool = True, keep_history: bool = False) -> dict:
        from src.detection.keypoint_extractor import draw_overlay
        t = time.perf_counter() if t is None else t
        t0 = time.perf_counter()
        reuse = self._frame_i % self.detect_interval != 0
        res = self.extractor.process(frame_bgr, dt_ms=dt_ms, reuse_person=reuse)
        self._frame_i += 1
        if keep_history:
            self.history_feats.append(res.features)
        events = self.decoder.update(t, res.features, res.any_hand)
        new_sentence = None
        for ev in events:
            if ev.kind == "sign":
                self.all_signs.append(ev.label)
                self.confidences.append(ev.confidence)
                log.info("Ký hiệu: %s (%.2f)", ev.label, ev.confidence)
            elif ev.kind == "sentence":
                new_sentence = self._finish_sentence(ev.signs)
        dt = time.perf_counter() - t0
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6)) if self._fps else 1.0 / max(dt, 1e-6)
        return {
            "frame": draw_overlay(frame_bgr, res) if draw else frame_bgr,
            "current": self.decoder.current,
            "signs": list(self.all_signs),
            "display_signs": [self.translator.display(s) for s in self.all_signs],
            "text": " ".join(self.sentences),
            "new_sentence": new_sentence,
            "hands": res.hands_present,
            "fps": self._fps,
        }

    def _finish_sentence(self, labels: list[str]) -> str | None:
        tr = self.translator.translate(labels)
        if not tr.text:
            return None
        self.sentences.append(tr.text)
        log.info("Câu: %s", tr.text)
        if self.tts is not None and self.cfg.tts.auto_speak:
            self.speak_async(tr.text)
        return tr.text

    def flush(self) -> str | None:
        """Kết thúc câu đang dở (khi dừng nhận diện / hết video)."""
        if self.decoder.signs:
            return self._finish_sentence(self.decoder.end_sentence().signs)
        return None

    def speak_async(self, text: str | None = None, on_error=None) -> threading.Thread | None:
        text = text if text is not None else " ".join(self.sentences)
        if not text or self.tts is None:
            return None

        def run():
            try:
                self.tts.speak(text)
                self.tts_error = None
            except Exception as exc:
                self.tts_error = str(exc)
                log.error("TTS lỗi: %s", exc)
                if on_error:
                    on_error(str(exc))
        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th

    def clear(self) -> None:
        self.decoder.reset()
        self.all_signs.clear(); self.sentences.clear(); self.confidences.clear(); self.history_feats.clear()
        self.extractor.reset()

    def close(self) -> None:
        self.extractor.close()
        if self.tts is not None:
            self.tts.stop()


def recognize_video_file(cfg, checkpoint: Path | None, video: Path, device: str | None = None,
                         speak: bool = False, recognizer: RealtimeRecognizer | None = None) -> dict:
    """Nhận diện một video có sẵn: giải mã liên tục + dự đoán cả video (cho video một ký hiệu)."""
    from src.preprocessing.video_loader import iter_frames
    own = recognizer is None
    rec = recognizer or RealtimeRecognizer(cfg, checkpoint, device, enable_tts=speak)
    rec.clear()
    fps = cfg.video.target_fps
    dt_ms = int(1000 / fps)
    n = 0
    t_start = time.perf_counter()
    try:
        for t, frame in iter_frames(video, fps, cfg.video.max_side, cfg.video.min_brightness,
                                    cfg.video.min_std, tuple(cfg.video.extensions)):
            rec.process_frame(frame, t=t, dt_ms=dt_ms, draw=False, keep_history=True)
            n += 1
        rec.flush()
        whole = None
        if len(rec.history_feats) >= 2:
            whole = rec.classifier.predict(np.stack(rec.history_feats), cfg.inference.top_k)
        # Video ngắn chứa một ký hiệu: nếu giải mã liên tục không bắt được gì, dùng dự đoán cả video
        if not rec.all_signs and whole is not None and whole.accepted:
            rec.all_signs.append(whole.label)
            rec.confidences.append(whole.confidence)
            tr = rec.translator.translate([whole.label])
            if tr.text:
                rec.sentences.append(tr.text)
        elapsed = time.perf_counter() - t_start
        text = " ".join(rec.sentences)
        result = {"video": str(video), "frames": n, "processing_fps": n / elapsed if elapsed else 0.0,
                  "signs": list(rec.all_signs), "display_signs": [rec.translator.display(s) for s in rec.all_signs],
                  "confidences": list(rec.confidences), "text": text,
                  "whole_video": None if whole is None else {"label": whole.label, "confidence": whole.confidence,
                                                             "top_k": whole.top_k, "accepted": whole.accepted}}
        if speak and text and rec.tts is not None:
            result["audio"] = str(rec.tts.speak(text, block=True))
        return result
    finally:
        if own:
            rec.close()


# ---------------------------------------------------------------------------- CLI webcam
def put_text_vi(img: np.ndarray, text: str, org, size=24, color=(255, 255, 255)) -> np.ndarray:
    """cv2.putText không hiển thị dấu tiếng Việt → vẽ bằng Pillow nếu có font."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = None
        for cand in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"):
            if Path(cand).is_file():
                font = ImageFont.truetype(cand, size)
                break
        if font is None:
            raise OSError
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        x, y = org
        d.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
        d.text((x, y), text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size / 32, color, 2)
        return img


def run_webcam(cfg, camera: int = 0, checkpoint: Path | None = None) -> None:
    import platform
    rec = RealtimeRecognizer(cfg, checkpoint)
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        rec.close()
        raise RuntimeError(f"Không mở được webcam {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    log.info("Webcam đang chạy. Phím: q = thoát, c = xóa, s = đọc lại câu.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("Mất frame từ webcam")
                break
            from src.preprocessing.video_loader import resize_max_side
            frame = resize_max_side(frame, cfg.video.max_side)
            st = rec.process_frame(frame)   # xử lý ảnh KHÔNG lật
            view = cv2.flip(st["frame"], 1)  # chỉ lật để hiển thị như gương
            cur = st["current"]
            line1 = f"{cur.label} {cur.confidence:.2f}" if cur else "..."
            view = put_text_vi(view, f"FPS {st['fps']:.1f} | {line1}", (10, 10), 22, (0, 255, 0))
            view = put_text_vi(view, " ".join(f"[{s}]" for s in st["display_signs"][-6:]), (10, 40), 22)
            view = put_text_vi(view, st["text"][-60:], (10, view.shape[0] - 40), 26, (0, 255, 255))
            cv2.imshow("Vietnamese Sign Language", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                rec.clear()
            if k == ord("s"):
                rec.speak_async()
    finally:
        rec.flush()
        cap.release()
        cv2.destroyAllWindows()
        rec.close()


def main() -> None:
    from src.utils import load_config, setup_logging
    ap = argparse.ArgumentParser(description="Nhận diện ngôn ngữ ký hiệu từ webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.project.log_level)
    run_webcam(cfg, args.camera, Path(args.checkpoint) if args.checkpoint else None)


if __name__ == "__main__":
    main()
