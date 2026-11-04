"""Xem nhanh cấu trúc một thư mục dataset (không sửa gì).

    python tools/inspect_dataset.py arichve
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def describe_dir(d: Path, indent: str = "") -> None:
    files = [f for f in d.rglob("*") if f.is_file()]
    exts = Counter(f.suffix.lower() or "(không đuôi)" for f in files)
    subdirs = [p for p in d.iterdir() if p.is_dir()]
    print(f"{indent}{d.name}/  {len(subdirs)} thư mục con, {len(files)} file  | đuôi: "
          + ", ".join(f"{e}×{n}" for e, n in exts.most_common(6)))
    if subdirs:
        sizes = sorted(((len([f for f in s.rglob('*') if f.is_file()]), s.name) for s in subdirs), reverse=True)
        counts = [n for n, _ in sizes]
        print(f"{indent}   file/thư mục con: nhiều nhất {counts[0]}, ít nhất {counts[-1]}, "
              f"trung bình {sum(counts) / len(counts):.1f}")
        print(f"{indent}   vd thư mục con: {[n for _, n in sizes[:5]]}")
        sample = next((s for s in subdirs if any(s.iterdir())), None)
        if sample:
            print(f"{indent}   vd file trong '{sample.name}': {[f.name for f in sorted(sample.iterdir())[:6]]}")
    else:
        print(f"{indent}   vd file: {[f.name for f in sorted(files)[:6]]}")


def show_table(f: Path) -> None:
    print(f"\n== {f}  ({f.stat().st_size / 1024:.0f} KB)")
    if f.suffix.lower() == ".json":
        data = json.loads(f.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            items = list(data.items())
            print(f"   dict, {len(items)} khóa. Vd: {items[:5]}")
        elif isinstance(data, list):
            print(f"   list, {len(data)} phần tử. Vd: {data[:3]}")
    elif f.suffix.lower() in (".csv", ".tsv", ".txt"):
        lines = f.read_bytes().decode("utf-8-sig", errors="replace").splitlines()
        print(f"   {len(lines)} dòng:")
        for line in lines[:5]:
            print("   ", line[:200])
    elif f.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(f)
        print(f"   {len(df)} dòng, cột: {list(df.columns)}")
        print(df.head(5).to_string())


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not root.is_dir():
        sys.exit(f"Không thấy {root}")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        print()
        describe_dir(d)
        for s in sorted(p for p in d.iterdir() if p.is_dir()):
            describe_dir(s, "    ")
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".json", ".csv", ".tsv", ".xlsx", ".xls") and f.stat().st_size < 50e6:
            show_table(f)
    # file đặc trưng (npy) — in hình dạng
    for f in list(root.rglob("*.npy"))[:2]:
        import numpy as np
        a = np.load(f, allow_pickle=False)
        print(f"\n== {f.name}: shape {a.shape}, dtype {a.dtype}")


if __name__ == "__main__":
    main()
