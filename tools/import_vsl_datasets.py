"""Gộp ViSignLanguage-Video (Hugging Face) + VSL QIPEDC (Kaggle) vào dataset/ đúng định dạng dự án.

    python tools/import_vsl_datasets.py --visign hf_vsl/train --qipedc D:/qipedc
    python tools/import_vsl_datasets.py --visign hf_vsl/train --qipedc D:/qipedc --max-classes 30

Quy tắc:
- ViSignLanguage-Video là nguồn CHÍNH → chia train/val/test phân tầng theo nhãn (mặc định 70/15/15).
- QIPEDC (thường 1 video/từ) chỉ được thêm vào TRAIN, và chỉ cho các từ có trong ViSign
  → tăng đa dạng người ký mà không làm sai lệch kết quả đánh giá.
- Tên nhãn tiếng Việt có dấu → thư mục không dấu (xin_chao); nghĩa có dấu ghi vào configs/labels.yaml.
- File đích đặt tên <nhan>__<nguon>__<so>.mp4, đường dẫn toàn ASCII (OpenCV trên Windows lỗi với đường dẫn có dấu).
- Clip không đọc được hoặc quá ít frame bị bỏ qua; mọi quyết định ghi vào runs/import_report.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
REGION_SUFFIX = {"B": "bac", "N": "nam", "T": "trung"}


# ----------------------------------------------------------------------------- chuẩn hóa nhãn
def norm_text(s: str) -> str:
    """So khớp nhãn giữa hai bộ: NFC, chữ thường, bỏ phần (bắc)/(nam)/(trung), gọn khoảng trắng."""
    s = unicodedata.normalize("NFC", str(s)).lower().strip()
    s = re.sub(r"\((?:miền\s*)?(bắc|nam|trung)\)", " ", s)
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFC", norm_text(s)).replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s or "nhan"


def count_frames(path: Path) -> int:
    """Đếm frame đọc được. Đọc qua bản sao ASCII nếu đường dẫn có ký tự Unicode (lỗi OpenCV/Windows)."""
    src = str(path)
    tmp = None
    if not src.isascii():
        tmp = ROOT / "runs" / "_probe_tmp" / f"probe{path.suffix.lower()}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, tmp)
        src = str(tmp)
    cap = cv2.VideoCapture(src)
    n = 0
    try:
        if cap.isOpened():
            while n < 2000:
                ok, fr = cap.read()
                if not ok:
                    break
                if fr is not None and fr.size:
                    n += 1
    finally:
        cap.release()
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return n


# ----------------------------------------------------------------------------- đọc ViSign
class _SafeUnpickler(__import__("pickle").Unpickler):
    """Chỉ cho phép kiểu dữ liệu cơ bản — không chạy code lạ trong file .pkl."""
    _OK = {("builtins", n) for n in ("dict", "list", "tuple", "set", "str", "int", "float")} | \
          {("collections", "OrderedDict"), ("collections", "defaultdict")}

    def find_class(self, module, name):
        if (module, name) in self._OK:
            return getattr(__import__(module), name)
        if module.startswith("numpy"):  # số nguyên numpy (np.int64...) trong mapping
            import numpy
            if name in ("dtype", "scalar", "_reconstruct", "ndarray") or module.endswith("multiarray"):
                mod = __import__(module, fromlist=[name])
                return getattr(mod, name)
        raise __import__("pickle").UnpicklingError(f"Từ chối kiểu '{module}.{name}' trong label_mapping.pkl")


def load_label_mapping(path: Path) -> dict[str, str]:
    """label_mapping.pkl → {tên_thư_mục/id (str): tên nhãn tiếng Việt}. Tự nhận cả hai chiều."""
    with path.open("rb") as f:
        m = _SafeUnpickler(f).load()
    if isinstance(m, (list, tuple)):
        m = {i: v for i, v in enumerate(m)}
    if not isinstance(m, dict):
        sys.exit(f"label_mapping.pkl có kiểu không hỗ trợ: {type(m)}")
    def is_id(x):
        return isinstance(x, int) or (hasattr(x, "item") and isinstance(x.item(), int)) or str(x).strip().isdigit()
    # dạng {tên: id} → đảo lại thành {id: tên}
    if m and all(is_id(v) for v in m.values()) and not all(is_id(k) for k in m.keys()):
        m = {v: k for k, v in m.items()}
    out = {str(int(k) if is_id(k) else k).strip(): unicodedata.normalize("NFC", str(v)) for k, v in m.items()}
    print(f"[ViSign] label_mapping.pkl: {len(out)} nhãn, vd: {list(out.items())[:3]}")
    return out


def load_visign(root: Path, mapping_path: Path | None = None) -> dict[str, list[Path]]:
    """root/<nhãn>/*.mp4  (thư mục 'train' sau khi giải nén dataset.zip)."""
    if not root.is_dir():
        sys.exit(f"Không thấy thư mục ViSign: {root}")
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.lower() == "train":  # người dùng trỏ vào thư mục cha
        root = subdirs[0]
    if mapping_path is None:
        for cand in (root / "label_mapping.pkl", root.parent / "label_mapping.pkl"):
            if cand.is_file():
                mapping_path = cand
                break
    mapping = load_label_mapping(mapping_path) if mapping_path else {}
    out: dict[str, list[Path]] = {}
    unmapped = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        vids = sorted(f for f in d.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
        if not vids:
            continue
        key = d.name.strip()
        key_id = str(int(key)) if key.isdigit() else key
        name = mapping.get(key_id) or mapping.get(key)
        if name is None:
            if key.isdigit():
                unmapped.append(key)
            name = key
        out.setdefault(unicodedata.normalize("NFC", name), []).extend(vids)
    if unmapped:
        print(f"⚠ {len(unmapped)} thư mục là số nhưng không có trong label_mapping.pkl: {unmapped[:10]} "
              "→ nhãn sẽ hiển thị là con số. Dùng --label-mapping để chỉ file ánh xạ.")
    if not out:
        sys.exit(f"{root} không có thư mục nhãn chứa video.")
    return out


# ----------------------------------------------------------------------------- đọc QIPEDC
def _read_table(path: Path) -> list[dict]:
    suf = path.suffix.lower()
    if suf in (".csv", ".tsv", ".txt"):
        raw = path.read_bytes()
        for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        return list(csv.DictReader(text.splitlines(), dialect=dialect))
    if suf in (".xlsx", ".xls"):
        import pandas as pd
        return pd.read_excel(path, dtype=str).fillna("").to_dict("records")
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            return [{"id": k, "label": v} for k, v in data.items()]
        return data
    sys.exit(f"Không hỗ trợ định dạng file nhãn: {path.name}")


def load_qipedc(root: Path, id_col: str | None, label_col: str | None) -> dict[str, list[tuple[Path, str]]]:
    """Trả về {nhãn_gốc: [(video, vùng)]}. vùng = bac/nam/trung/'' theo hậu tố B/N/T của mã video."""
    lab_dir, vid_dir = root / "Labels", root / "Videos"
    if not vid_dir.is_dir():
        sys.exit(f"Không thấy {vid_dir} (cần thư mục Labels/ và Videos/ của bộ QIPEDC)")
    tables = [p for p in lab_dir.iterdir() if p.is_file()] if lab_dir.is_dir() else []
    if not tables:
        sys.exit(f"Không thấy file nhãn trong {lab_dir}")
    rows = _read_table(tables[0])
    if not rows:
        sys.exit(f"File nhãn {tables[0].name} rỗng")
    cols = list(rows[0].keys())
    videos = {f.stem.upper(): f for f in vid_dir.rglob("*") if f.suffix.lower() in VIDEO_EXTS}

    def content_score(c: str, want_id: bool) -> int:
        vals = [str(r.get(c, "")) for r in rows[:300]]
        if want_id:  # cột mã video = cột có nhiều giá trị trùng tên file video nhất
            return sum(Path(v.strip()).stem.upper() in videos for v in vals)
        return sum(bool(re.search(r"[^\x00-\x7f]", v)) for v in vals)  # có chữ tiếng Việt

    def pick(cands, want_id: bool):
        scores = {c: content_score(c, want_id) for c in cols}
        best = max(cols, key=lambda c: scores[c])
        if want_id and scores[best] > 0:
            return best  # khớp theo nội dung đáng tin hơn tên cột (vd cột ID=1,2,3 không phải tên file)
        for c in cols:
            if c.strip().lower() in cands:
                return c
        return best

    id_col = id_col or pick({"id", "video", "video_id", "videoid", "file", "filename", "ma", "mã", "code"}, True)
    label_col = label_col or pick({"label", "labels", "gloss", "word", "nhan", "nhãn", "tu", "từ", "meaning"}, False)
    print(f"[QIPEDC] file nhãn: {tables[0].name} | cột mã video: '{id_col}' | cột nhãn: '{label_col}' "
          f"| {len(rows)} dòng, {len(videos)} video")
    if id_col == label_col:
        sys.exit("Không tự xác định được cột. Dùng --qipedc-id-col và --qipedc-label-col. Các cột: " + str(cols))

    out: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    miss = 0
    for r in rows:
        vid = Path(str(r.get(id_col, "")).strip()).stem.upper()
        label = str(r.get(label_col, "")).strip()
        if not vid or not label:
            continue
        f = videos.get(vid)
        if f is None:
            miss += 1
            continue
        m = re.match(r"^(D?\d+)([BNT])$", vid)
        region = REGION_SUFFIX[m.group(2)] if m else ""
        out[unicodedata.normalize("NFC", label)].append((f, region))
    if miss:
        print(f"[QIPEDC] {miss} dòng nhãn không có video tương ứng (bỏ qua)")
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visign", required=True, help="thư mục train/ của ViSignLanguage-Video (đã giải nén)")
    ap.add_argument("--label-mapping", default=None,
                    help="label_mapping.pkl (mặc định: tự tìm cạnh thư mục --visign)")
    ap.add_argument("--qipedc", default=None, help="thư mục gốc bộ QIPEDC (chứa Labels/ và Videos/)")
    ap.add_argument("--qipedc-id-col", default=None)
    ap.add_argument("--qipedc-label-col", default=None)
    ap.add_argument("--regions", default="all",
                    help="biến thể vùng QIPEDC cần lấy: all | none | bac,nam,trung (none = chỉ video không hậu tố)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-per-class", type=int, default=8, help="nhãn ViSign ít clip hợp lệ hơn sẽ bị bỏ")
    ap.add_argument("--max-classes", type=int, default=0, help="chỉ lấy N nhãn nhiều clip nhất (0 = tất cả)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="xóa dataset/train|val|test cũ trước khi ghi")
    ap.add_argument("--no-check", action="store_true", help="không đếm frame (nhanh hơn, kém an toàn)")
    args = ap.parse_args()

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    existing = [p for s in ("train", "val", "test") for p in (out / s).glob("*/*") if p.is_file()]
    if existing and not args.overwrite:
        sys.exit(f"{out} đã có {len(existing)} video. Thêm --overwrite để xóa và ghi lại.")
    if args.overwrite:
        for s in ("train", "val", "test"):
            if (out / s).is_dir():
                shutil.rmtree(out / s)
            (out / s).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    report: list[dict] = []

    # 1) ViSign: lọc clip lỗi
    visign = load_visign(Path(args.visign), Path(args.label_mapping) if args.label_mapping else None)
    print(f"[ViSign] {len(visign)} nhãn, {sum(map(len, visign.values()))} clip")
    valid: dict[str, list[Path]] = {}
    for i, (label, vids) in enumerate(visign.items(), 1):
        good = []
        for v in vids:
            n = args.min_frames if args.no_check else count_frames(v)
            ok = n >= args.min_frames
            if ok:
                good.append(v)
            else:
                report.append({"source": "visign", "label": label, "file": str(v), "split": "",
                               "status": f"bỏ: {n} frame"})
        valid[label] = good
        if i % 10 == 0:
            print(f"  kiểm tra {i}/{len(visign)} nhãn")

    # 2) gộp nhãn trùng sau chuẩn hóa, bỏ nhãn quá ít clip
    by_slug: dict[str, dict] = {}
    for label, vids in valid.items():
        slug = slugify(label)
        entry = by_slug.setdefault(slug, {"text": norm_text(label), "display": label, "vids": []})
        entry["vids"] += vids
    dropped = {s: e for s, e in by_slug.items() if len(e["vids"]) < args.min_per_class}
    for s, e in dropped.items():
        print(f"  bỏ nhãn '{e['display']}' ({len(e['vids'])} clip < {args.min_per_class})")
    by_slug = {s: e for s, e in by_slug.items() if s not in dropped}
    if args.max_classes:
        keep = sorted(by_slug, key=lambda s: -len(by_slug[s]["vids"]))[:args.max_classes]
        by_slug = {s: by_slug[s] for s in sorted(keep)}
    if len(by_slug) < 2:
        sys.exit("Còn dưới 2 nhãn hợp lệ — kiểm tra lại đường dẫn --visign.")

    # 3) chia ViSign phân tầng
    counts = defaultdict(lambda: defaultdict(int))
    for slug, e in by_slug.items():
        vids = e["vids"][:]
        rng.shuffle(vids)
        n = len(vids)
        n_te, n_va = max(1, round(n * args.test)), max(1, round(n * args.val))
        for k, v in enumerate(vids):
            split = "test" if k < n_te else "val" if k < n_te + n_va else "train"
            dst = out / split / slug / f"{slug}__unknown__{k:04d}{v.suffix.lower()}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(v, dst)
            counts[slug][split] += 1
            report.append({"source": "visign", "label": e["display"], "file": str(v), "split": split,
                           "status": f"ok → {dst.relative_to(out)}"})

    # 4) QIPEDC → chỉ train, chỉ nhãn khớp
    matched_q = 0
    if args.qipedc:
        regions = {"all": None, "none": {""}}.get(args.regions)
        if regions is None and args.regions != "all":
            regions = {r.strip() for r in args.regions.split(",")} | {""}
        q = load_qipedc(Path(args.qipedc), args.qipedc_id_col, args.qipedc_label_col)
        text_to_slug = {e["text"]: s for s, e in by_slug.items()}
        for qlabel, items in q.items():
            slug = text_to_slug.get(norm_text(qlabel))
            if slug is None:
                continue
            for f, region in items:
                if regions is not None and region not in regions:
                    continue
                if not args.no_check and count_frames(f) < args.min_frames:
                    report.append({"source": "qipedc", "label": qlabel, "file": str(f), "split": "",
                                   "status": "bỏ: lỗi/ít frame"})
                    continue
                tag = f"qipedc{region}" if region else "qipedc"
                dst = out / "train" / slug / f"{slug}__{tag}__{f.stem.lower()}{f.suffix.lower()}"
                shutil.copy2(f, dst)
                counts[slug]["qipedc"] += 1
                matched_q += 1
                report.append({"source": "qipedc", "label": qlabel, "file": str(f), "split": "train",
                               "status": f"ok → {dst.relative_to(out)}"})
        print(f"[QIPEDC] thêm {matched_q} video vào train cho "
              f"{sum(1 for s in counts if counts[s]['qipedc'])}/{len(by_slug)} nhãn")

    # 5) labels.yaml (giữ nhãn cũ, thêm/ghi đè nhãn mới)
    lab_path = ROOT / "configs" / "labels.yaml"
    labels = yaml.safe_load(lab_path.read_text(encoding="utf-8")) if lab_path.is_file() else {}
    labels = labels or {}
    for slug, e in by_slug.items():
        labels[slug] = norm_text(e["display"])
    lab_path.write_text("# Ánh xạ nhãn → tiếng Việt (tự sinh bởi tools/import_vsl_datasets.py, có thể sửa tay)\n" +
                        yaml.safe_dump(labels, allow_unicode=True, sort_keys=True), encoding="utf-8")

    # 6) báo cáo
    rep = ROOT / "runs" / "import_report.csv"
    rep.parent.mkdir(parents=True, exist_ok=True)
    with rep.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "label", "file", "split", "status"])
        w.writeheader()
        w.writerows(report)
    print(f"\n{'nhãn':28s} {'train':>5s} {'+qip':>5s} {'val':>4s} {'test':>5s}  nghĩa")
    for slug in sorted(counts):
        c = counts[slug]
        print(f"{slug:28s} {c['train']:5d} {c['qipedc']:5d} {c['val']:4d} {c['test']:5d}  {labels[slug]}")
    tot = lambda k: sum(c[k] for c in counts.values())
    print(f"{'TỔNG':28s} {tot('train'):5d} {tot('qipedc'):5d} {tot('val'):4d} {tot('test'):5d}")
    print(f"\nĐã ghi {out}, {lab_path.relative_to(ROOT)}, {rep.relative_to(ROOT)}")
    print("Tiếp theo: python train.py check  →  python train.py all")


if __name__ == "__main__":
    main()
