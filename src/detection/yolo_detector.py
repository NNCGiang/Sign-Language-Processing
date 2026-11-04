"""YOLO26 — phát hiện người ký hiệu + khung xương cơ thể, và (tùy chọn) bàn tay.

YOLO26-pose pretrained (COCO) cho 17 khớp cơ thể/người. Nó KHÔNG có điểm ngón tay,
nên bàn tay được lấy bằng:
  * MediaPipe HandLandmarker chạy trên ROI người do YOLO26 cắt ra (mặc định), hoặc
  * YOLO26-pose fine-tune trên bộ Hand Keypoints (21 điểm/tay) — xem `train_hand_pose()`.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Chỉ số COCO-17
COCO_KPTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10
UPPER_BODY_IDX = list(range(13))  # mũi → hông: phần cơ thể dùng trong ký hiệu


@dataclass
class PersonDetection:
    box: np.ndarray        # (4,) x1,y1,x2,y2 theo pixel
    conf: float
    keypoints: np.ndarray  # (17,3) x,y theo pixel + độ tin cậy


@dataclass
class HandDetection:
    box: np.ndarray
    conf: float
    keypoints: np.ndarray  # (21,3) x,y pixel + conf


def resolve_weights(name_or_path: str, yolo_dir: Path) -> Path | str:
    """Ưu tiên file trong models/yolo26/. Nếu là trọng số chuẩn của Ultralytics thì tự tải về đó."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    local = Path(yolo_dir) / p.name
    if local.is_file():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        got = Path(attempt_download_asset(str(local)))
        if got.is_file():
            if got.resolve() != local.resolve():
                shutil.move(str(got), local)
            return local
    except Exception as exc:  # trọng số tùy chỉnh chưa tồn tại, mạng lỗi...
        log.debug("attempt_download_asset lỗi: %s", exc)
    raise FileNotFoundError(
        f"Không tìm thấy trọng số YOLO '{name_or_path}'. Đặt file vào {local} "
        "hoặc kiểm tra kết nối mạng (Ultralytics tự tải trọng số chuẩn từ GitHub).")


class YoloPoseDetector:
    """Bao bọc mô hình YOLO26-pose của Ultralytics."""

    def __init__(self, weights: str, yolo_dir: Path, device: str = "cpu",
                 conf: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO
        self.weights = resolve_weights(weights, yolo_dir)
        self.model = YOLO(str(self.weights))
        if getattr(self.model, "task", "pose") != "pose":
            raise ValueError(f"{self.weights} không phải mô hình pose (task={self.model.task}).")
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        log.info("YOLO26 pose: %s trên %s", Path(self.weights).name, device)

    def _predict(self, frame_bgr: np.ndarray):
        return self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                  verbose=False)[0]

    def detect_all(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        r = self._predict(frame_bgr)
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kd = r.keypoints.data.cpu().numpy()
        if kd.shape[-1] == 2:  # một số bản xuất không có cột conf
            kd = np.concatenate([kd, np.ones(kd.shape[:-1] + (1,), kd.dtype)], -1)
        return [(boxes[i], float(confs[i]), kd[i]) for i in range(len(boxes))]


class PersonDetector(YoloPoseDetector):
    """YOLO26-pose COCO: chọn người ký hiệu chính (khung lớn nhất × độ tin cậy)."""

    def detect(self, frame_bgr: np.ndarray) -> PersonDetection | None:
        dets = self.detect_all(frame_bgr)
        if not dets:
            return None

        def score(d):
            b = d[0]
            return (b[2] - b[0]) * (b[3] - b[1]) * d[1]
        box, conf, kp = max(dets, key=score)
        return PersonDetection(box.astype(np.float32), conf, kp.astype(np.float32))


class YoloHandDetector(YoloPoseDetector):
    """YOLO26-pose đã fine-tune trên Hand Keypoints (21 điểm). Trả về tối đa 2 tay."""

    def detect(self, frame_bgr: np.ndarray) -> list[HandDetection]:
        dets = self.detect_all(frame_bgr)
        dets = [d for d in dets if d[2].shape[0] == 21]
        dets.sort(key=lambda d: d[1], reverse=True)
        return [HandDetection(b.astype(np.float32), c, k.astype(np.float32)) for b, c, k in dets[:2]]


def train_hand_pose(base_weights: str, yolo_dir: Path, out_name: str = "yolo26n-hand-pose.pt",
                    data: str = "hand-keypoints.yaml", epochs: int = 100, imgsz: int = 640,
                    batch: int = 16, device: str = "auto", project: Path | None = None,
                    workers: int = 0) -> Path:
    """Fine-tune YOLO26-pose để phát hiện bàn tay + 21 điểm ngón tay.

    `data` mặc định là bộ Hand Keypoints của Ultralytics (tự tải ~370 MB lần đầu).
    Có thể thay bằng file YAML của bộ dữ liệu bàn tay tự gán nhãn (định dạng YOLO-pose,
    kpt_shape: [21, 3]).
    """
    from ultralytics import YOLO
    from src.utils import select_device
    dev = select_device(device)
    weights = resolve_weights(base_weights, yolo_dir)
    model = YOLO(str(weights))
    project = Path(project) if project else Path(yolo_dir).parent.parent / "runs" / "yolo_hands"
    log.info("Fine-tune %s trên %s (%d epoch, device=%s)", Path(weights).name, data, epochs, dev)
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch,
                device=dev.replace("cuda:", "") if dev.startswith("cuda") else "cpu",
                project=str(project), name="train", exist_ok=True, workers=workers, patience=20)
    best = project / "train" / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Huấn luyện xong nhưng không thấy {best}")
    out = Path(yolo_dir) / out_name
    shutil.copy2(best, out)
    log.info("Đã lưu trọng số YOLO26 bàn tay: %s", out)
    metrics = model.val(data=data, imgsz=imgsz, device=dev.replace("cuda:", "") if dev.startswith("cuda") else "cpu")
    try:
        log.info("Pose mAP50-95 (val): %.4f", float(metrics.pose.map))
    except Exception:
        pass
    return out
