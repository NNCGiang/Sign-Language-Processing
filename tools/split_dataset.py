"""Chia thư mục video thô (raw/<nhãn>/*.mp4) thành dataset/train|val|test.

    python tools/split_dataset.py dataset_raw dataset --by-signer
    python tools/split_dataset.py dataset_raw dataset --test-signers nguoi05 --val-signers nguoi04

--by-signer : mỗi NGƯỜI chỉ thuộc một tập (đánh giá trung thực khả năng tổng quát cho người mới).
              Tên file phải dạng <nhãn>__<người>__<số>.mp4 (tools/record_dataset.py tự đặt như vậy).
Mặc định    : chia ngẫu nhiên phân tầng theo nhãn (70/15/15).
File được SAO CHÉP, không di chuyển.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

EXTS = {".mp4", ".avi", ".mov"}


def signer_of(p: Path) -> str:
    parts = p.stem.split("__")
    return parts[1] if len(parts) >= 3 else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--by-signer", action="store_true")
    ap.add_argument("--val-signers", nargs="*", default=None)
    ap.add_argument("--test-signers", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw, out = Path(args.raw_dir), Path(args.out_dir)
    if not raw.is_dir():
        sys.exit(f"Không thấy {raw}")
    files = defaultdict(list)
    for label_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        for f in sorted(label_dir.iterdir()):
            if f.suffix.lower() in EXTS:
                files[label_dir.name].append(f)
    if not files:
        sys.exit("Không có video nào")
    rng = random.Random(args.seed)
    assign: dict[Path, str] = {}

    if args.by_signer or args.val_signers or args.test_signers:
        signers = sorted({signer_of(f) for fs in files.values() for f in fs})
        if "unknown" in signers:
            sys.exit("Có file không theo mẫu <nhãn>__<người>__<số>.mp4 — không chia theo người được.")
        if len(signers) < 3 and not (args.val_signers and args.test_signers):
            sys.exit(f"Cần ≥3 người để chia theo người (hiện có {signers}).")
        test_s = set(args.test_signers or [])
        val_s = set(args.val_signers or [])
        rest = [s for s in signers if s not in test_s | val_s]
        rng.shuffle(rest)
        if not test_s:
            test_s = set(rest[:max(1, round(len(signers) * args.test))]); rest = rest[len(test_s):]
        if not val_s:
            val_s = set(rest[:max(1, round(len(signers) * args.val))]); rest = rest[len(val_s):]
        if not rest:
            sys.exit("Không còn người nào cho tập train.")
        for fs in files.values():
            for f in fs:
                s = signer_of(f)
                assign[f] = "test" if s in test_s else "val" if s in val_s else "train"
        print(f"Train: {sorted(rest)}\nVal: {sorted(val_s)}\nTest: {sorted(test_s)}")
    else:
        for fs in files.values():
            fs = fs[:]
            rng.shuffle(fs)
            n = len(fs)
            n_te, n_va = max(1, round(n * args.test)), max(1, round(n * args.val))
            if n - n_te - n_va < 1:
                sys.exit(f"Nhãn {fs[0].parent.name} có quá ít video ({n}) để chia.")
            for i, f in enumerate(fs):
                assign[f] = "test" if i < n_te else "val" if i < n_te + n_va else "train"

    if out.exists() and any(out.glob("*/*/*")) and not args.overwrite:
        sys.exit(f"{out} đã có dữ liệu. Thêm --overwrite để ghi tiếp.")
    counts = defaultdict(lambda: defaultdict(int))
    for f, split in assign.items():
        dst = out / split / f.parent.name / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        counts[f.parent.name][split] += 1
    print(f"{'nhãn':20s} train  val  test")
    for label in sorted(counts):
        c = counts[label]
        print(f"{label:20s} {c['train']:5d} {c['val']:4d} {c['test']:5d}")


if __name__ == "__main__":
    main()
