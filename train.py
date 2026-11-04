"""Điểm vào huấn luyện.

    python train.py check        # kiểm tra dataset (định dạng, số lượng, trùng người train/test)
    python train.py extract      # video → đặc trưng YOLO26 + tay + mặt (.npy, có cache)
    python train.py train        # huấn luyện BiGRU/BiLSTM, lưu checkpoint tốt nhất
    python train.py all          # check → extract → train → evaluate (tập test)
    python train.py yolo-hands   # fine-tune YOLO26-pose cho bàn tay (21 điểm)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, resolve_path, select_device, set_seed, setup_logging

log = logging.getLogger("train")


def cmd_check(cfg) -> dict:
    from src.preprocessing.video_loader import validate_dataset
    rep = validate_dataset(resolve_path(cfg.paths.dataset_dir), tuple(cfg.video.extensions), cfg.video.min_frames)
    log.info("Tổng số video: %d", rep["total"])
    for split, counts in rep["counts"].items():
        log.info("[%s] %d nhãn, %d video, người ký hiệu: %s", split, len(counts), sum(counts.values()),
                 ", ".join(rep["signers"].get(split, [])))
        for label, n in sorted(counts.items()):
            log.info("    %-20s %d", label, n)
    for path, err in rep["bad"]:
        log.error("Video lỗi: %s — %s", path, err)
    for w in rep["warnings"]:
        log.warning(w)
    if rep["total"] == 0:
        log.error("Dataset rỗng. Xem README mục 'Thu thập dữ liệu' hoặc chạy tools/record_dataset.py")
    return rep


def cmd_extract(cfg, device: str, overwrite: bool = False) -> pd.DataFrame:
    from src.detection.keypoint_extractor import KeypointExtractor
    from src.recognition.feature_sequence import extract_dataset
    ext = KeypointExtractor(cfg, device)
    try:
        df = extract_dataset(cfg, ext, resolve_path(cfg.paths.dataset_dir),
                             resolve_path(cfg.paths.features_dir), overwrite)
    finally:
        ext.close()
    ok = df[df.status == "ok"]
    log.info("Trích xong: %d/%d video hợp lệ.", len(ok), len(df))
    log.info("\n%s", ok.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_string())
    return df


def _run_epoch(model, loader, criterion, device, optimizer=None):
    import torch
    train = optimizer is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            n += len(y)
    return tot_loss / max(n, 1), correct / max(n, 1)


def cmd_train(cfg, device: str) -> Path:
    import torch
    from torch.utils.data import DataLoader
    from src.recognition.feature_sequence import SignSequenceDataset, model_input_dim
    from src.recognition.lstm_model import SignRecognitionModel
    from src.recognition.sign_classifier import CHECKPOINT_NAME, save_checkpoint

    set_seed(cfg.project.seed)
    fdir = resolve_path(cfg.paths.features_dir)
    mpath = fdir / "manifest.csv"
    if not mpath.is_file():
        raise FileNotFoundError(f"Thiếu {mpath}. Chạy trước: python train.py extract")
    manifest = pd.read_csv(mpath)
    ok = manifest[manifest.status == "ok"]
    labels = sorted(ok[ok.split == "train"].label.unique())
    if len(labels) < 2:
        raise RuntimeError("Cần ít nhất 2 nhãn có video huấn luyện hợp lệ.")
    l2i = {l: i for i, l in enumerate(labels)}
    T, vel = cfg.sequence.length, cfg.sequence.use_velocity
    aug = cfg.training.augment if cfg.training.augment.enabled else None
    train_ds = SignSequenceDataset(manifest, "train", l2i, T, vel, aug, cfg.project.seed)
    val_ds = SignSequenceDataset(manifest, "val", l2i, T, vel)
    if len(val_ds) == 0:
        raise RuntimeError("Tập validation rỗng. Cần video trong dataset/val/<nhãn>/ "
                           "(hoặc dùng tools/split_dataset.py để chia).")
    log.info("Nhãn (%d): %s", len(labels), labels)
    log.info("Train: %d mẫu | Val: %d mẫu | thiết bị: %s", len(train_ds), len(val_ds), device)

    bs, nw = cfg.training.batch_size, cfg.training.num_workers
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=len(train_ds) > bs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    model = SignRecognitionModel.from_config(cfg.model, model_input_dim(vel), len(labels)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Mô hình %s: %.2fM tham số", cfg.model.rnn_type.upper(), n_params / 1e6)

    weight = None
    if cfg.training.class_weights:
        counts = np.bincount(train_ds.targets, minlength=len(labels)).astype(np.float32)
        weight = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.training.label_smoothing)
    val_criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.01)

    ckpt_dir = resolve_path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / CHECKPOINT_NAME
    history, best_loss, best_epoch, bad = [], float("inf"), 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_dl, criterion, device, opt)
        va_loss, va_acc = _run_epoch(model, val_dl, val_criterion, device)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": opt.param_groups[0]["lr"]})
        improved = va_loss < best_loss - 1e-4
        if improved:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            save_checkpoint(best_path, model, labels, cfg,
                            {"epoch": epoch, "val_loss": va_loss, "val_acc": va_acc})
        else:
            bad += 1
        if epoch == 1 or epoch % 5 == 0 or improved:
            log.info("Epoch %3d | train loss %.4f acc %.3f | val loss %.4f acc %.3f %s",
                     epoch, tr_loss, tr_acc, va_loss, va_acc, "★" if improved else "")
        if bad >= cfg.training.patience:
            log.info("Early stopping ở epoch %d (val loss không giảm %d epoch).", epoch, bad)
            break
    pd.DataFrame(history).to_csv(ckpt_dir / "history.csv", index=False)
    save_checkpoint(ckpt_dir / "last_model.pt", model, labels, cfg, {"epoch": epoch})
    with (ckpt_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    log.info("Huấn luyện xong sau %.1fs. Tốt nhất: epoch %d, val loss %.4f → %s",
             time.perf_counter() - t0, best_epoch, best_loss, best_path)
    from src.evaluation.evaluate import plot_history
    out = resolve_path(cfg.paths.runs_dir) / "training"
    out.mkdir(parents=True, exist_ok=True)
    plot_history(ckpt_dir / "history.csv", out)
    return best_path


def cmd_yolo_hands(cfg, args) -> None:
    from src.detection.yolo_detector import train_hand_pose
    train_hand_pose(cfg.yolo.pose_weights, resolve_path(cfg.paths.yolo_dir), cfg.hands.yolo_hand_weights,
                    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=cfg.yolo.device, project=resolve_path(cfg.paths.runs_dir) / "yolo_hands",
                    workers=cfg.training.num_workers)
    log.info("Để dùng: đặt hands.backend: yolo trong configs/config.yaml, rồi chạy lại extract + train.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "extract", "train", "all", "yolo-hands"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--overwrite", action="store_true", help="trích lại đặc trưng dù đã có cache")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--data", default="hand-keypoints.yaml", help="[yolo-hands] YAML dữ liệu bàn tay")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.project.log_level, resolve_path(cfg.paths.runs_dir) / "logs" / "train.log")
    device = select_device(cfg.yolo.device)
    try:
        if args.command == "check":
            rep = cmd_check(cfg)
            return 0 if rep["total"] and not rep["bad"] else 1
        if args.command == "extract":
            cmd_extract(cfg, device, args.overwrite)
        elif args.command == "train":
            if args.epochs:
                cfg["training"]["epochs"] = args.epochs
            cmd_train(cfg, device)
        elif args.command == "all":
            if args.epochs:
                cfg["training"]["epochs"] = args.epochs
            if cmd_check(cfg)["total"] == 0:
                return 1
            cmd_extract(cfg, device, args.overwrite)
            best = cmd_train(cfg, device)
            from src.evaluation.evaluate import evaluate_test
            evaluate_test(cfg, best, resolve_path(cfg.paths.runs_dir) / "evaluation", device)
        elif args.command == "yolo-hands":
            args.epochs = args.epochs or 100
            cmd_yolo_hands(cfg, args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
