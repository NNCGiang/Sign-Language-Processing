"""Chuyển chuỗi nhãn ký hiệu (gloss) → câu tiếng Việt.

Cách làm (minh bạch, có thể kiểm soát):
1. Bỏ nhãn nền, gộp nhãn lặp liên tiếp.
2. Khớp cụm dài nhất trong configs/grammar.yaml (xử lý thứ tự từ / phủ định / câu hỏi theo VSL).
3. Phần không khớp: ghép nghĩa từng nhãn theo configs/labels.yaml.
4. Thêm dấu câu (. ! ?) và viết hoa đầu câu.

Không đoán nghĩa: nhãn không có trong từ điển được giữ nguyên dạng đọc được và đánh dấu `unknown`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    text: str
    sentences: list[str]
    glosses: list[str]
    display_glosses: list[str]
    unknown: list[str] = field(default_factory=list)


def _cap(s: str) -> str:
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


class VietnameseTranslator:
    def __init__(self, labels_file: str | Path | None = None, grammar_file: str | Path | None = None,
                 background_labels=("_nen",)):
        from src.utils import load_yaml
        self.lexicon: dict[str, str] = {}
        self.phrases: list[tuple[tuple[str, ...], str, str]] = []
        self.question_labels: set[str] = set()
        self.exclamation_labels: set[str] = set()
        self.background = set(background_labels or [])
        if labels_file and Path(labels_file).is_file():
            self.lexicon = {str(k): ("" if v is None else str(v)) for k, v in (load_yaml(labels_file) or {}).items()}
        if grammar_file and Path(grammar_file).is_file():
            g = load_yaml(grammar_file) or {}
            for p in g.get("phrases", []) or []:
                gloss = tuple(str(x) for x in p["gloss"])
                self.phrases.append((gloss, str(p["text"]), str(p.get("end", "") or "")))
            self.phrases.sort(key=lambda t: -len(t[0]))
            self.question_labels = set(g.get("question_labels", []) or [])
            self.exclamation_labels = set(g.get("exclamation_labels", []) or [])

    @classmethod
    def from_config(cls, cfg) -> "VietnameseTranslator":
        from src.utils import resolve_path
        return cls(resolve_path(cfg.paths.labels_file), resolve_path(cfg.paths.grammar_file),
                   cfg.inference.background_labels)

    def word(self, label: str) -> str:
        if label in self.lexicon:
            return self.lexicon[label]
        return label.replace("_", " ").strip()

    def display(self, label: str) -> str:
        return self.word(label).upper() or label

    def translate(self, glosses: list[str]) -> TranslationResult:
        seq: list[str] = []
        for g in glosses:
            if g in self.background or (g in self.lexicon and self.lexicon[g] == ""):
                continue
            if not seq or seq[-1] != g:
                seq.append(g)
        unknown = [g for g in seq if g not in self.lexicon]
        sentences: list[str] = []
        chunk: list[str] = []

        def flush():
            if not chunk:
                return
            words = " ".join(self.word(g) for g in chunk).strip()
            if words:
                if any(g in self.question_labels for g in chunk):
                    end = "?"
                elif all(g in self.exclamation_labels for g in chunk):
                    end = "!"
                else:
                    end = "."
                sentences.append(_cap(words) + end)
            chunk.clear()

        i = 0
        while i < len(seq):
            match = None
            for gloss, text, end in self.phrases:
                n = len(gloss)
                if tuple(seq[i:i + n]) == gloss:
                    match = (n, text, end)
                    break
            if match:
                n, text, end = match
                if end:
                    flush()
                    sentences.append(_cap(text) + end)
                else:
                    chunk.append("__phrase__:" + text)
                    self.lexicon.setdefault("__phrase__:" + text, text)
                i += n
            else:
                chunk.append(seq[i])
                i += 1
        flush()
        text = " ".join(sentences)
        return TranslationResult(text, sentences, seq, [self.display(g) for g in seq], unknown)
