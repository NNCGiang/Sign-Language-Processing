"""Tiện ích dùng chung: cấu hình, logging, đường dẫn, seed, thiết bị, tải file."""
from __future__ import annotations

import logging
import random
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Cấu hình logging ra console (UTF-8) và file (tùy chọn)."""
    handlers: list[logging.Handler] = []
    stream = logging.StreamHandler(sys.stdout)
    try:  # tránh lỗi in tiếng Việt trên console Windows cũ
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    handlers.append(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=_LOG_FORMAT, handlers=handlers, force=True)
    # bớt log ồn ào của thư viện
    for noisy in ("ultralytics", "matplotlib", "PIL", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Config(dict):
    """dict cho phép truy cập dạng thuộc tính: cfg.training.lr"""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(f"Thiếu khóa cấu hình: '{item}'") from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        def conv(v):
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v
        return conv(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _wrap(data)


def load_yaml(path: str | Path) -> Any:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    """Đường dẫn tương đối → tính từ thư mục gốc dự án."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def select_device(pref: str | int = "auto") -> str:
    """Trả về chuỗi thiết bị cho PyTorch / Ultralytics."""
    import torch
    pref = str(pref).lower()
    if pref == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if pref.isdigit():
        pref = f"cuda:{pref}"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA không khả dụng, chuyển sang CPU.")
        return "cpu"
    return pref


def download_file(url: str, dest: Path, timeout: int = 120) -> Path:
    """Tải file nếu chưa có. Ném lỗi rõ ràng kèm hướng dẫn tải thủ công."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log = logging.getLogger(__name__)
    log.info("Đang tải %s → %s", url, dest)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Không tải được {url}: {exc}\n"
            f"Hãy tải thủ công file này bằng trình duyệt và đặt vào: {dest}"
        ) from exc
    return dest
