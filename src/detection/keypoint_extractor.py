"""Trích xuất vector đặc trưng mỗi frame: thân (YOLO26) + 2 bàn tay + khuôn mặt.

Bố cục vector (FEATURE_DIM = 291):
  [0:39]    thân trên: 13 khớp COCO × (x, y, conf)          — từ YOLO26-pose
  [39:144]  tay TRÁI của người ký hiệu: 21×(x,y,z) theo thân + 21×(x,y) hình dạng cục bộ
  [144:249] tay PHẢI: như trên
  [249:251] cờ có tay (trái, phải)
  [251:291] 20 điểm khuôn mặt × (x, y)                       — MediaPipe FaceLandmarker

Chuẩn hóa: gốc tọa độ = trung điểm hai vai, đơn vị = khoảng cách hai vai.
→ không phụ thuộc vị trí đứng, khoảng cách tới camera, độ phân giải.
Thiếu dữ liệu (không thấy tay/mặt) → 0 và cờ = 0.

Quan trọng: frame phải ở dạng KHÔNG lật gương (cả khi huấn luyện lẫn webcam),
để tay trái/phải nhất quán.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .yolo_detector import (L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, UPPER_BODY_IDX,
                            PersonDetection, PersonDetector, YoloHandDetector)

log = logging.getLogger(__name__)

N_BODY = len(UPPER_BODY_IDX)
N_HAND = 21
# Điểm mặt chọn lọc (chỉ số FaceMesh 478): lông mày, mắt, mũi, môi, cằm — mang biểu cảm
FACE_IDX = [70, 105, 107, 336, 334, 300,      # lông mày
            159, 145, 386, 374,               # mí mắt trên/dưới
            1, 152,                           # chóp mũi, cằm
            61, 291, 0, 17, 13, 14, 78, 308]  # môi
N_FACE = len(FACE_IDX)

BODY_DIM = N_BODY * 3
HAND_DIM = N_HAND * 3 + N_HAND * 2
FACE_DIM = N_FACE * 2
FEATURE_DIM = BODY_DIM + 2 * HAND_DIM + 2 + FACE_DIM

SLICES = {
    "body": slice(0, BODY_DIM),
    "left_hand": slice(BODY_DIM, BODY_DIM + HAND_DIM),
    "right_hand": slice(BODY_DIM + HAND_DIM, BODY_DIM + 2 * HAND_DIM),
    "hand_flags": slice(BODY_DIM + 2 * HAND_DIM, BODY_DIM + 2 * HAND_DIM + 2),
    "face": slice(BODY_DIM + 2 * HAND_DIM + 2, FEATURE_DIM),
}

# Cạnh nối để vẽ khung xương
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]
BODY_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (0, 5), (0, 6)]


@dataclass
class FrameResult:
    features: np.ndarray                   # (FEATURE_DIM,)
    person: PersonDetection | None = None
    hands_px: list[np.ndarray | None] = field(default_factory=lambda: [None, None])  # [trái, phải] (21,3) pixel
    face_px: np.ndarray | None = None      # (N_FACE,2) pixel

    @property
    def hands_present(self) -> tuple[bool, bool]:
        return self.hands_px[0] is not None, self.hands_px[1] is not None

    @property
    def any_hand(self) -> bool:
        return any(self.hands_present)


class _MediaPipeHands:
    def __init__(self, model_path: Path, det: float, pres: float, track: float):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=det, min_hand_presence_confidence=pres,
            min_tracking_confidence=track)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[tuple[np.ndarray, str, float]]:
        """Trả về [(21×3 tọa độ chuẩn hóa theo ảnh vào, 'Left'/'Right', score)]."""
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(img, ts_ms)
        out = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            cat = hd[0] if hd else None
            out.append((arr, cat.category_name if cat else "", float(cat.score) if cat else 0.0))
        return out

    def close(self):
        self.landmarker.close()


class _MediaPipeFace:
    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray | None:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.landmarker.detect_for_video(img, ts_ms)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        return np.array([[lms[i].x, lms[i].y] for i in FACE_IDX], dtype=np.float32)

    def close(self):
        self.landmarker.close()


def feature_signature(cfg) -> dict:
    """Mô tả cấu hình đặc trưng — lưu trong checkpoint để kiểm tra tương thích khi suy luận."""
    return {"feature_dim": FEATURE_DIM, "hands_backend": cfg.hands.backend,
            "face_enabled": bool(cfg.face.enabled), "body_idx": UPPER_BODY_IDX, "face_idx": FACE_IDX}


class KeypointExtractor:
    """Kết hợp YOLO26 (người + thân) với bộ phát hiện tay và mặt → vector đặc trưng/frame."""

    def __init__(self, cfg, device: str = "cpu"):
        from src.utils import download_file, resolve_path
        self.cfg = cfg
        yolo_dir = resolve_path(cfg.paths.yolo_dir)
        mp_dir = resolve_path(cfg.paths.mediapipe_dir)
        self.person = PersonDetector(cfg.yolo.pose_weights, yolo_dir, device, cfg.yolo.conf, cfg.yolo.imgsz)
        self.hand_backend = str(cfg.hands.backend).lower()
        self.mp_hands = self.yolo_hands = self.mp_face = None
        if self.hand_backend == "mediapipe":
            path = download_file(cfg.hands.mediapipe_url, mp_dir / cfg.hands.mediapipe_model)
            self.mp_hands = _MediaPipeHands(path, cfg.hands.min_detection_confidence,
                                            cfg.hands.min_presence_confidence,
                                            cfg.hands.min_tracking_confidence)
        elif self.hand_backend == "yolo":
            self.yolo_hands = YoloHandDetector(cfg.hands.yolo_hand_weights, yolo_dir, device,
                                               cfg.hands.yolo_hand_conf, cfg.yolo.imgsz)
        elif self.hand_backend != "none":
            raise ValueError(f"hands.backend không hợp lệ: {cfg.hands.backend}")
        else:
            log.warning("hands.backend = none → KHÔNG có đặc trưng ngón tay. Chỉ dùng để thử pipeline.")
        if cfg.face.enabled:
            path = download_file(cfg.face.mediapipe_url, mp_dir / cfg.face.mediapipe_model)
            self.mp_face = _MediaPipeFace(path)
        self._ts_ms = 0  # MediaPipe VIDEO mode yêu cầu timestamp tăng dần
        self._last_person: PersonDetection | None = None
        self._frame_idx = 0

    # ------------------------------------------------------------------ helpers
    def _next_ts(self, step_ms: int = 33) -> int:
        self._ts_ms += max(1, step_ms)
        return self._ts_ms

    def _roi(self, frame: np.ndarray, person: PersonDetection | None) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        if person is None or not self.cfg.hands.use_person_roi:
            return 0, 0, w, h
        x1, y1, x2, y2 = person.box
        m = self.cfg.hands.roi_margin
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = max(0, int(x1 - m * bw)), min(w, int(x2 + m * bw))
        y1, y2 = max(0, int(y1 - m * bh)), min(h, int(y2 + m * bh))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return 0, 0, w, h
        return x1, y1, x2, y2

    @staticmethod
    def _normalizer(person: PersonDetection | None, shape) -> tuple[np.ndarray, float]:
        h, w = shape[:2]
        if person is not None:
            k = person.keypoints
            if k[L_SHOULDER, 2] > 0.3 and k[R_SHOULDER, 2] > 0.3:
                c = (k[L_SHOULDER, :2] + k[R_SHOULDER, :2]) / 2
                s = float(np.linalg.norm(k[L_SHOULDER, :2] - k[R_SHOULDER, :2]))
                if s > 8:
                    return c, s
            x1, y1, x2, y2 = person.box
            bh = max(y2 - y1, 1.0)
            return np.array([(x1 + x2) / 2, y1 + 0.3 * bh], np.float32), 0.25 * bh
        return np.array([w / 2, h / 2], np.float32), 0.25 * h

    @staticmethod
    def _assign_slots(hands: list[tuple[np.ndarray, str]], person: PersonDetection | None) -> list[np.ndarray | None]:
        """Gán tay vào [trái, phải] của người ký hiệu. Ưu tiên khoảng cách tới cổ tay YOLO26."""
        slots: list[np.ndarray | None] = [None, None]
        if not hands:
            return slots
        k = person.keypoints if person is not None else None
        if k is not None and k[L_WRIST, 2] > 0.3 and k[R_WRIST, 2] > 0.3:
            wl, wr = k[L_WRIST, :2], k[R_WRIST, :2]
            if len(hands) == 1:
                p = hands[0][0][0, :2]
                slots[0 if np.linalg.norm(p - wl) <= np.linalg.norm(p - wr) else 1] = hands[0][0]
            else:
                a, b = hands[0][0], hands[1][0]
                d_ab = np.linalg.norm(a[0, :2] - wl) + np.linalg.norm(b[0, :2] - wr)
                d_ba = np.linalg.norm(b[0, :2] - wl) + np.linalg.norm(a[0, :2] - wr)
                slots = [a, b] if d_ab <= d_ba else [b, a]
            return slots
        # Không có cổ tay: dùng nhãn handedness của MediaPipe (giả định ảnh gương →
        # "Left" của MediaPipe trên ảnh không lật là tay PHẢI của người), hoặc vị trí x.
        for arr, side in hands:
            if side in ("Left", "Right"):
                idx = 1 if side == "Left" else 0
            else:  # YOLO tay: tay ở bên phải ảnh là tay trái của người (người nhìn vào camera)
                idx = 0 if arr[0, 0] >= (k[0, 0] if k is not None else arr[0, 0]) else 1
            if slots[idx] is None:
                slots[idx] = arr
            elif slots[1 - idx] is None:
                slots[1 - idx] = arr
        return slots

    # ------------------------------------------------------------------ main API
    def process(self, frame_bgr: np.ndarray, dt_ms: int = 33, reuse_person: bool = False) -> FrameResult:
        """Xử lý một frame BGR (không lật gương) → FrameResult."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame rỗng")
        h, w = frame_bgr.shape[:2]
        if reuse_person and self._last_person is not None:
            person = self._last_person
        else:
            person = self.person.detect(frame_bgr)
            self._last_person = person
        center, scale = self._normalizer(person, frame_bgr.shape)
        ts = self._next_ts(dt_ms)

        feat = np.zeros(FEATURE_DIM, np.float32)
        # --- thân
        if person is not None:
            kp = person.keypoints[UPPER_BODY_IDX].copy()
            ok = kp[:, 2] > 0.2
            body = np.zeros((N_BODY, 3), np.float32)
            body[ok, :2] = (kp[ok, :2] - center) / scale
            body[:, 2] = kp[:, 2]
            feat[SLICES["body"]] = body.reshape(-1)

        # --- tay
        hands_px: list[tuple[np.ndarray, str]] = []
        x1, y1, x2, y2 = self._roi(frame_bgr, person)
        if self.mp_hands is not None:
            crop = frame_bgr[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            cw, ch = x2 - x1, y2 - y1
            for arr, side, _ in self.mp_hands.detect(rgb, ts):
                px = np.empty((N_HAND, 3), np.float32)
                px[:, 0] = arr[:, 0] * cw + x1
                px[:, 1] = arr[:, 1] * ch + y1
                px[:, 2] = arr[:, 2] * cw  # z cùng thang đo với x (pixel)
                hands_px.append((px, side))
        elif self.yolo_hands is not None:
            crop = frame_bgr[y1:y2, x1:x2]
            for hd in self.yolo_hands.detect(crop):
                px = hd.keypoints.copy()
                px[:, 0] += x1
                px[:, 1] += y1
                px[:, 2] = 0.0
                hands_px.append((px, ""))
        slots = self._assign_slots(hands_px, person)
        for i, (name, arr) in enumerate(zip(("left_hand", "right_hand"), slots)):
            if arr is None:
                continue
            glob = np.empty((N_HAND, 3), np.float32)
            glob[:, :2] = (arr[:, :2] - center) / scale
            glob[:, 2] = arr[:, 2] / scale
            palm = float(np.linalg.norm(arr[9, :2] - arr[0, :2])) or 1.0
            local = (arr[:, :2] - arr[0, :2]) / max(palm, 1e-3)
            feat[SLICES[name]] = np.concatenate([glob.reshape(-1), local.reshape(-1)])
            feat[SLICES["hand_flags"].start + i] = 1.0

        # --- mặt
        face_px = None
        if self.mp_face is not None:
            fx1, fy1, fx2, fy2 = x1, y1, x2, y2
            crop = frame_bgr[fy1:fy2, fx1:fx2]
            res = self.mp_face.detect(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), ts)
            if res is not None:
                face_px = np.stack([res[:, 0] * (fx2 - fx1) + fx1, res[:, 1] * (fy2 - fy1) + fy1], 1)
                feat[SLICES["face"]] = ((face_px - center) / scale).reshape(-1)

        self._frame_idx += 1
        return FrameResult(feat, person, list(slots), face_px)

    def reset(self) -> None:
        """Gọi giữa hai video để không dùng lại ROI cũ (timestamp vẫn tăng tiếp)."""
        self._last_person = None
        self._ts_ms += 1000

    def close(self) -> None:
        for m in (self.mp_hands, self.mp_face):
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass


def draw_overlay(frame: np.ndarray, res: FrameResult) -> np.ndarray:
    """Vẽ khung người, khung xương, bàn tay, điểm mặt lên frame (trả về bản sao)."""
    out = frame.copy()
    if res.person is not None:
        x1, y1, x2, y2 = res.person.box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 200, 80), 2)
        k = res.person.keypoints
        for a, b in BODY_EDGES:
            if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                cv2.line(out, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), (255, 180, 0), 2)
    colors = [(0, 140, 255), (255, 0, 200)]  # trái: cam, phải: tím
    for arr, col in zip(res.hands_px, colors):
        if arr is None:
            continue
        for a, b in HAND_EDGES:
            cv2.line(out, tuple(arr[a, :2].astype(int)), tuple(arr[b, :2].astype(int)), col, 2)
        for p in arr:
            cv2.circle(out, tuple(p[:2].astype(int)), 3, col, -1)
    if res.face_px is not None:
        for p in res.face_px:
            cv2.circle(out, tuple(p.astype(int)), 2, (0, 255, 255), -1)
    return out
