"""Đánh giá mô hình trên tập TEST độc lập.

Chỉ số: Accuracy, Precision, Recall, F1 (macro & weighted), Confusion Matrix, Loss,
FPS trích đặc trưng, độ trễ suy luận; WER/CER cho video câu liên tục (nếu có file tham chiếu).

Chạy:
    python -m src.evaluation.evaluate                       # đánh giá tập test
    python -m src.evaluation.evaluate --sentences refs.csv  # thêm WER/CER (cột: video,reference)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- WER / CER
def edit_distance(ref: list, hyp: list) -> int:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[-1]


def _norm_text(s: str) -> str:
    import re
    return " ".join(re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def wer(refs: list[str], hyps: list[str]) -> float:
    errs = sum(edit_distance(_norm_text(r).split(), _norm_text(h).split()) for r, h in zip(refs, hyps))
    total = sum(len(_norm_text(r).split()) for r in refs)
    return errs / max(total, 1)


def cer(refs: list[str], hyps: list[str]) -> float:
    errs = sum(edit_distance(list(_norm_text(r).replace(" ", "")), list(_norm_text(h).replace(" ", "")))
               for r, h in zip(refs, hyps))
    total = sum(len(_norm_text(r).replace(" ", "")) for r in refs)
    return errs / max(total, 1)


# ----------------------------------------------------------------------------- biểu đồ
def plot_history(history_csv: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.read_csv(history_csv)
    out = []
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_loss"], label="Training loss")
    ax.plot(h["epoch"], h["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p = out_dir / "loss_curve.png"; fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_acc"], label="Training accuracy")
    ax.plot(h["epoch"], h["val_acc"], label="Validation accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02); ax.set_title("Accuracy")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p = out_dir / "accuracy_curve.png"; fig.savefig(p, dpi=130); plt.close(fig); out.append(p)
    return out


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, normalize: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = cm.astype(float)
    if normalize:
        m = m / np.maximum(m.sum(1, keepdims=True), 1)
    n = len(labels)
    size = max(5, 0.45 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title("Confusion matrix (test)")
    if n <= 25:
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if m[i, j] > .5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_latency(stats: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, vals = [], []
    for k, label in (("feature_ms_per_frame", "Trích đặc trưng\n(ms/frame)"),
                     ("classifier_ms_per_window", "Phân loại GRU/LSTM\n(ms/cửa sổ)")):
        if stats.get(k) is not None:
            names.append(label); vals.append(stats[k])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(names, vals, color=["#4C78A8", "#F58518"][:len(vals)])
    ax.bar_label(bars, fmt="%.1f ms")
    title = "Độ trễ suy luận"
    if stats.get("pipeline_fps"):
        title += f" — pipeline ≈ {stats['pipeline_fps']:.1f} FPS ({stats.get('device', '')})"
    ax.set_title(title, fontsize=10); ax.set_ylabel("ms"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


# ----------------------------------------------------------------------------- đánh giá chính
def evaluate_test(cfg, checkpoint: Path, out_dir: Path, device: str, measure_video_fps: int = 5) -> dict:
    import torch
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    from src.recognition.feature_sequence import SignSequenceDataset
    from src.recognition.sign_classifier import SignClassifier
    from src.utils import resolve_path

    out_dir.mkdir(parents=True, exist_ok=True)
    clf = SignClassifier(checkpoint, device, cfg.inference.confidence_threshold,
                         cfg.inference.background_labels, cfg)
    manifest_path = resolve_path(cfg.paths.features_dir) / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Thiếu {manifest_path}. Chạy: python train.py extract")
    manifest = pd.read_csv(manifest_path)
    l2i = {l: i for i, l in enumerate(clf.labels)}
    ds = SignSequenceDataset(manifest, "test", l2i, clf.seq_len, clf.use_velocity)
    if len(ds) == 0:
        raise RuntimeError("Tập test rỗng — cần video trong dataset/test/<nhãn>/ (không dùng lại dữ liệu train).")
    unknown = set(manifest[manifest.split == "test"].label) - set(l2i)
    if unknown:
        log.warning("Test có nhãn mô hình chưa học (bỏ qua): %s", sorted(unknown))

    xs = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    ys = np.array(ds.targets)
    with torch.no_grad():
        logits = clf.model(xs)
        loss = float(torch.nn.functional.cross_entropy(logits, torch.from_numpy(ys).to(device)))
        probs = torch.softmax(logits, -1).cpu().numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)

    # độ trễ bộ phân loại: từng cửa sổ một (giống thời gian thực)
    one = xs[:1]
    with torch.no_grad():
        for _ in range(5):
            clf.model(one)
        t0 = time.perf_counter(); n_rep = 50
        for _ in range(n_rep):
            clf.model(one)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    cls_ms = (time.perf_counter() - t0) / n_rep * 1000

    labels_present = sorted(set(ys) | set(pred))
    names = [clf.labels[i] for i in labels_present]
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(ys, pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(ys, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(ys, pred, labels=labels_present)
    thr = clf.threshold
    accepted = conf >= thr
    report = classification_report(ys, pred, labels=labels_present, target_names=names, zero_division=0, digits=4)

    results = {
        "n_test": int(len(ys)), "n_classes": len(clf.labels), "loss": loss,
        "accuracy": float(accuracy_score(ys, pred)),
        "precision_macro": float(p_ma), "recall_macro": float(r_ma), "f1_macro": float(f_ma),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "confidence_threshold": thr,
        "coverage_at_threshold": float(accepted.mean()),
        "accuracy_at_threshold": float((pred[accepted] == ys[accepted]).mean()) if accepted.any() else None,
        "classifier_ms_per_window": cls_ms, "device": device,
    }

    # FPS trích đặc trưng đo trên một vài video test thật
    if measure_video_fps > 0:
        from src.detection.keypoint_extractor import KeypointExtractor
        from src.preprocessing.video_loader import iter_frames
        vids = manifest[(manifest.split == "test") & (manifest.status == "ok")]["video"].head(measure_video_fps)
        ext = KeypointExtractor(cfg, device)
        n, t = 0, 0.0
        try:
            for v in vids:
                frames = [f for _, f in iter_frames(v, cfg.video.target_fps, cfg.video.max_side)]
                ext.reset()
                if frames:
                    ext.process(frames[0])  # khởi động
                t0 = time.perf_counter()
                for f in frames[1:]:
                    ext.process(f)
                t += time.perf_counter() - t0
                n += max(len(frames) - 1, 0)
        finally:
            ext.close()
        if n:
            feat_ms = t / n * 1000
            results["feature_ms_per_frame"] = feat_ms
            stride = max(1, int(cfg.inference.stride_frames))
            results["pipeline_fps"] = 1000.0 / (feat_ms + cls_ms / stride)

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(cm, index=names, columns=names).to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8")
    plot_confusion(cm, names, out_dir / "confusion_matrix.png")
    plot_latency(results, out_dir / "latency.png")
    hist = checkpoint.parent / "history.csv"
    if hist.is_file():
        plot_history(hist, out_dir)
    # danh sách mẫu sai để kiểm tra lại
    wrong = [{"features": str(ds.items[i][0]), "true": clf.labels[ys[i]], "pred": clf.labels[pred[i]],
              "confidence": round(float(conf[i]), 4)} for i in np.where(pred != ys)[0]]
    pd.DataFrame(wrong).to_csv(out_dir / "misclassified.csv", index=False, encoding="utf-8")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("\n%s", report)
    log.info("TEST — acc=%.4f  F1(macro)=%.4f  loss=%.4f  phân loại %.2f ms/cửa sổ",
             results["accuracy"], results["f1_macro"], loss, cls_ms)
    if "pipeline_fps" in results:
        log.info("Trích đặc trưng %.1f ms/frame → pipeline ≈ %.1f FPS",
                 results["feature_ms_per_frame"], results["pipeline_fps"])
    return results


def evaluate_sentences(cfg, checkpoint: Path, refs_csv: Path, out_dir: Path, device: str) -> dict:
    """WER/CER trên video câu liên tục. CSV gồm cột: video, reference (câu tiếng Việt đúng)."""
    from src.realtime.webcam_recognition import recognize_video_file
    df = pd.read_csv(refs_csv)
    if not {"video", "reference"} <= set(df.columns):
        raise ValueError("File tham chiếu cần cột 'video' và 'reference'")
    hyps = []
    for v in df["video"]:
        res = recognize_video_file(cfg, checkpoint, Path(v), device=device, speak=False)
        hyps.append(res["text"])
    df["hypothesis"] = hyps
    out = {"wer": wer(list(df.reference), hyps), "cer": cer(list(df.reference), hyps), "n": len(df)}
    df.to_csv(out_dir / "sentence_predictions.csv", index=False, encoding="utf-8")
    with (out_dir / "sentence_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("Câu liên tục: WER=%.4f  CER=%.4f  (n=%d)", out["wer"], out["cer"], out["n"])
    return out


def main() -> None:
    from src.recognition.sign_classifier import CHECKPOINT_NAME
    from src.utils import load_config, resolve_path, select_device, setup_logging
    ap = argparse.ArgumentParser(description="Đánh giá mô hình trên tập test độc lập")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="runs/evaluation")
    ap.add_argument("--sentences", default=None, help="CSV (video,reference) để tính WER/CER")
    ap.add_argument("--fps-videos", type=int, default=5, help="số video test dùng đo FPS (0 = bỏ qua)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.project.log_level)
    device = select_device(cfg.yolo.device)
    ckpt = Path(args.checkpoint) if args.checkpoint else resolve_path(cfg.paths.checkpoints_dir) / CHECKPOINT_NAME
    out = resolve_path(args.out)
    evaluate_test(cfg, ckpt, out, device, args.fps_videos)
    if args.sentences:
        evaluate_sentences(cfg, ckpt, Path(args.sentences), out, device)
    log.info("Kết quả lưu tại %s", out)


if __name__ == "__main__":
    main()
