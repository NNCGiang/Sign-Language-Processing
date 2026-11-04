"""Giao diện PyQt5: tải video / webcam → nhận diện → văn bản tiếng Việt → phát âm.

    python -m app.main
"""
from __future__ import annotations

import logging
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QProgressBar, QPushButton, QTextEdit,
                             QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import load_config, resolve_path, setup_logging  # noqa: E402

log = logging.getLogger("app")


class RecognitionWorker(QObject):
    """Chạy trong QThread: đọc frame (webcam/video), nhận diện, gửi kết quả về giao diện."""
    frame_ready = pyqtSignal(QImage)
    state_ready = pyqtSignal(dict)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, recognizer, source, cfg, recognize: bool):
        super().__init__()
        self.rec, self.source, self.cfg = recognizer, source, cfg
        self.recognize = recognize
        self._running = True

    def stop(self):
        self._running = False

    def set_recognize(self, on: bool):
        self.recognize = on

    def run(self):
        from src.preprocessing.video_loader import resize_max_side
        is_cam = isinstance(self.source, int)
        if is_cam:
            backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        else:
            cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            self.error.emit(f"Không mở được {'webcam' if is_cam else self.source}")
            self.finished.emit("")
            return
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target = float(self.cfg.video.target_fps)
        step = 1.0 / target if (not is_cam and target < fps_src) else 0.0
        next_t, idx = 0.0, 0
        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                t = idx / fps_src if not is_cam else time.perf_counter()
                idx += 1
                if not is_cam and step and t + 1e-6 < next_t:
                    continue
                next_t = (next_t + step) if step else t
                if frame is None or frame.size == 0:
                    continue
                frame = resize_max_side(frame, self.cfg.video.max_side)
                if self.recognize and self.rec is not None:
                    dt_ms = int(1000 * (step or 1 / 30))
                    st = self.rec.process_frame(frame, t=t, dt_ms=dt_ms, keep_history=not is_cam)
                    view = st["frame"]
                    self.state_ready.emit(st)
                else:
                    view = frame
                if is_cam:
                    view = cv2.flip(view, 1)  # hiển thị như gương; xử lý vẫn dùng ảnh gốc
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                self.frame_ready.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
                if not is_cam and not self.recognize:
                    time.sleep(1 / target)
        except Exception as exc:
            log.exception("Lỗi khi nhận diện")
            self.error.emit(str(exc))
        finally:
            cap.release()
        final = ""
        if self.recognize and self.rec is not None:
            final = self.rec.flush() or ""
            if not is_cam and not self.rec.all_signs and len(self.rec.history_feats) >= 2:
                # video một ký hiệu: dùng dự đoán trên toàn bộ video
                pred = self.rec.classifier.predict(np.stack(self.rec.history_feats), self.cfg.inference.top_k)
                self.state_ready.emit({"current": pred, "whole": True})
                if pred.accepted:
                    self.rec.all_signs.append(pred.label)
                    tr = self.rec.translator.translate([pred.label])
                    if tr.text:
                        self.rec.sentences.append(tr.text)
                        final = tr.text
            self.state_ready.emit({"signs": list(self.rec.all_signs),
                                   "display_signs": [self.rec.translator.display(s) for s in self.rec.all_signs],
                                   "text": " ".join(self.rec.sentences)})
        self.finished.emit(final)


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.rec = None
        self.source = None           # đường dẫn video hoặc chỉ số webcam
        self.thread: QThread | None = None
        self.worker: RecognitionWorker | None = None
        self.setWindowTitle("Nhận diện & dịch Ngôn ngữ Ký hiệu Việt Nam — YOLO26 + GRU/LSTM")
        self.resize(1250, 760)
        self._build_ui()
        self._load_models()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        big = QFont(); big.setPointSize(13)
        self.video_label = QLabel("Chưa có video / webcam")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(720, 480)
        self.video_label.setStyleSheet("background:#111;color:#aaa;border-radius:6px;")

        self.btn_upload = QPushButton("📂 Tải video lên")
        self.btn_cam = QPushButton("📷 Bật webcam")
        self.btn_start = QPushButton("▶ Bắt đầu nhận diện")
        self.btn_stop = QPushButton("■ Dừng nhận diện")
        self.btn_speak = QPushButton("🔊 Phát âm tiếng Việt")
        self.btn_replay = QPushButton("↻ Phát lại âm thanh")
        self.btn_clear = QPushButton("✖ Xóa kết quả")
        for b in (self.btn_upload, self.btn_cam, self.btn_start, self.btn_stop,
                  self.btn_speak, self.btn_replay, self.btn_clear):
            b.setMinimumHeight(38)
        self.btn_upload.clicked.connect(self.on_upload)
        self.btn_cam.clicked.connect(self.on_webcam)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_speak.clicked.connect(self.on_speak)
        self.btn_replay.clicked.connect(self.on_replay)
        self.btn_clear.clicked.connect(self.on_clear)

        self.signs_box = QTextEdit(); self.signs_box.setReadOnly(True); self.signs_box.setFont(big)
        self.signs_box.setMaximumHeight(90)
        self.text_box = QTextEdit(); self.text_box.setReadOnly(True)
        f2 = QFont(); f2.setPointSize(16); f2.setBold(True); self.text_box.setFont(f2)
        self.current_label = QLabel("—"); self.current_label.setFont(big)
        self.conf_bar = QProgressBar(); self.conf_bar.setRange(0, 100); self.conf_bar.setFormat("%p%")
        self.topk_label = QLabel(""); self.topk_label.setWordWrap(True)
        self.status = QLabel("Sẵn sàng"); self.status.setWordWrap(True)

        g1 = QGroupBox("Ký hiệu nhận diện"); QVBoxLayout(g1).addWidget(self.signs_box)
        g2 = QGroupBox("Văn bản tiếng Việt"); QVBoxLayout(g2).addWidget(self.text_box)
        g3 = QGroupBox("Độ tin cậy (cửa sổ hiện tại)"); l3 = QVBoxLayout(g3)
        l3.addWidget(self.current_label); l3.addWidget(self.conf_bar); l3.addWidget(self.topk_label)

        controls = QGridLayout()
        controls.addWidget(self.btn_upload, 0, 0); controls.addWidget(self.btn_cam, 0, 1)
        controls.addWidget(self.btn_start, 1, 0); controls.addWidget(self.btn_stop, 1, 1)
        audio = QHBoxLayout(); audio.addWidget(self.btn_speak); audio.addWidget(self.btn_replay)

        right = QVBoxLayout()
        right.addLayout(controls); right.addWidget(g3); right.addWidget(g1); right.addWidget(g2, 1)
        right.addLayout(audio); right.addWidget(self.btn_clear); right.addWidget(self.status)

        root = QHBoxLayout()
        root.addWidget(self.video_label, 3)
        rw = QWidget(); rw.setLayout(right); rw.setMinimumWidth(420)
        root.addWidget(rw, 2)
        central = QWidget(); central.setLayout(root)
        self.setCentralWidget(central)
        self._set_running(False)

    def _load_models(self):
        from src.realtime.webcam_recognition import RealtimeRecognizer
        self.status.setText("Đang nạp YOLO26, MediaPipe và mô hình GRU/LSTM…")
        QApplication.processEvents()
        try:
            self.rec = RealtimeRecognizer(self.cfg)
            self.status.setText(f"Đã nạp mô hình ({len(self.rec.classifier.labels)} nhãn, thiết bị {self.rec.device}).")
        except Exception as exc:
            self.rec = None
            self.status.setText(f"⚠ Chưa nạp được mô hình: {exc}")
            QMessageBox.warning(self, "Chưa có mô hình",
                                f"{exc}\n\nVẫn có thể xem video/webcam, nhưng cần huấn luyện "
                                "(python train.py all) để nhận diện.")

    def _set_running(self, running: bool):
        self.btn_start.setEnabled(not running and self.source is not None and self.rec is not None)
        self.btn_stop.setEnabled(running)
        self.btn_upload.setEnabled(not running)
        self.btn_cam.setEnabled(not running)

    # ------------------------------------------------------------------ luồng xử lý
    def _launch(self, recognize: bool):
        self._stop_thread()
        self.thread = QThread(self)
        self.worker = RecognitionWorker(self.rec, self.source, self.cfg, recognize)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.state_ready.connect(self.on_state)
        self.worker.error.connect(lambda m: self.status.setText(f"⚠ {m}"))
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()
        self._set_running(recognize)

    def _stop_thread(self):
        if self.worker is not None:
            self.worker.stop()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
        self.thread = self.worker = None

    # ------------------------------------------------------------------ slots
    def on_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video ngôn ngữ ký hiệu", str(ROOT),
                                              "Video (*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV)")
        if not path:
            return
        from src.preprocessing.video_loader import VideoError, probe_video
        try:
            info = probe_video(path, tuple(self.cfg.video.extensions))
        except VideoError as exc:
            QMessageBox.critical(self, "Video không hợp lệ", str(exc))
            return
        self.source = Path(path)
        self.status.setText(f"Video: {self.source.name} — {info.duration:.1f}s, {info.fps:.0f} FPS, "
                            f"{info.width}×{info.height}. Bấm 'Bắt đầu nhận diện'.")
        self._show_first_frame(self.source)
        self._set_running(False)

    def _show_first_frame(self, path: Path):
        cap = cv2.VideoCapture(str(path)); ok, frame = cap.read(); cap.release()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
            self.on_frame(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())

    def on_webcam(self):
        self.source = 0
        self.status.setText("Webcam đang bật (xem trước). Bấm 'Bắt đầu nhận diện'.")
        self._launch(recognize=False)
        self.btn_start.setEnabled(self.rec is not None)
        self.btn_stop.setEnabled(True)

    def on_start(self):
        if self.rec is None or self.source is None:
            return
        if isinstance(self.source, Path):
            self.rec.clear()
        self.status.setText("Đang nhận diện…")
        if self.worker is not None and isinstance(self.source, int) and self.thread and self.thread.isRunning():
            self.worker.set_recognize(True)
            self._set_running(True)
        else:
            self._launch(recognize=True)

    def on_stop(self):
        self._stop_thread()
        self._set_running(False)
        self.status.setText("Đã dừng.")

    def on_frame(self, img: QImage):
        pix = QPixmap.fromImage(img).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def on_state(self, st: dict):
        cur = st.get("current")
        if cur is not None:
            name = self.rec.translator.display(cur.label) if self.rec else cur.label
            mark = "✓" if cur.accepted else "(dưới ngưỡng)"
            prefix = "Cả video: " if st.get("whole") else ""
            self.current_label.setText(f"{prefix}{name}  {mark}")
            self.conf_bar.setValue(int(round(cur.confidence * 100)))
            self.topk_label.setText("Top-k: " + ", ".join(f"{l} {p:.2f}" for l, p in cur.top_k))
        elif "fps" in st:
            self.current_label.setText("— (chưa thấy chuyển động tay)")
            self.conf_bar.setValue(0)
        if "display_signs" in st:
            self.signs_box.setPlainText(" ".join(f"[{s}]" for s in st["display_signs"]))
        if "text" in st:
            self.text_box.setPlainText(st["text"])
        if "fps" in st:
            hands = st.get("hands", (False, False))
            self.status.setText(f"Đang nhận diện… {st['fps']:.1f} FPS | tay trái: {'có' if hands[0] else '—'}"
                                f" | tay phải: {'có' if hands[1] else '—'}")

    def on_finished(self, final: str):
        self._set_running(False)
        if isinstance(self.source, Path):
            self.status.setText("Đã xử lý xong video." + ("" if self.rec and self.rec.all_signs
                                                         else " Không có ký hiệu đạt ngưỡng tin cậy."))

    def on_speak(self):
        text = self.text_box.toPlainText().strip()
        if not text or self.rec is None:
            self.status.setText("Chưa có văn bản để phát âm.")
            return
        self.status.setText("Đang tổng hợp giọng nói…")
        self.rec.speak_async(text, on_error=lambda m: None)

    def on_replay(self):
        if self.rec is None or self.rec.tts is None or self.rec.tts.last_audio is None:
            self.status.setText("Chưa có âm thanh để phát lại.")
            return
        try:
            self.rec.tts.replay()
        except Exception as exc:
            self.status.setText(f"⚠ {exc}")

    def on_clear(self):
        if self.rec is not None:
            self.rec.clear()
        self.signs_box.clear(); self.text_box.clear()
        self.current_label.setText("—"); self.conf_bar.setValue(0); self.topk_label.clear()
        self.status.setText("Đã xóa kết quả.")

    def closeEvent(self, e):
        self._stop_thread()
        if self.rec is not None:
            self.rec.close()
        super().closeEvent(e)


def main():
    cfg = load_config()
    setup_logging(cfg.project.log_level, resolve_path(cfg.paths.runs_dir) / "logs" / "app.log")
    app = QApplication(sys.argv)
    win = MainWindow(cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
