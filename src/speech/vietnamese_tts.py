"""Text-to-Speech tiếng Việt với nhiều engine dự phòng + cache + phát lại.

Engine (thử theo thứ tự trong config tts.engine_order):
  edge   — edge-tts, giọng neural vi-VN-HoaiMyNeural / vi-VN-NamMinhNeural (cần Internet, chất lượng tốt)
  gtts   — Google Translate TTS (cần Internet)
  vieneu — VieNeu-TTS chạy OFFLINE trên CPU (pip install vieneu)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


class VietnameseTTS:
    def __init__(self, engine_order=("edge", "gtts", "vieneu"), cache_dir: str | Path = "runs/audio",
                 edge_voice: str = "vi-VN-HoaiMyNeural", edge_rate: str = "+0%", gtts_lang: str = "vi",
                 vieneu_voice: str | None = None):
        self.engine_order = list(engine_order)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.edge_voice, self.edge_rate = edge_voice, edge_rate
        self.gtts_lang, self.vieneu_voice = gtts_lang, vieneu_voice
        self._vieneu = None
        self.last_audio: Path | None = None
        self._lock = threading.Lock()
        self._mixer_ready = False

    @classmethod
    def from_config(cls, cfg) -> "VietnameseTTS":
        from src.utils import resolve_path
        t = cfg.tts
        return cls(t.engine_order, resolve_path(t.cache_dir), t.edge_voice, t.edge_rate, t.gtts_lang,
                   t.get("vieneu_voice"))

    # ------------------------------------------------------------------ tổng hợp
    def _cache_path(self, engine: str, text: str, ext: str) -> Path:
        key = hashlib.md5(f"{engine}|{self.edge_voice}|{self.edge_rate}|{text}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{engine}_{key}.{ext}"

    def _edge(self, text: str) -> Path:
        import edge_tts
        out = self._cache_path("edge", text, "mp3")
        if not out.is_file():
            async def run():
                await edge_tts.Communicate(text, self.edge_voice, rate=self.edge_rate).save(str(out))
            asyncio.run(run())
        return out

    def _gtts(self, text: str) -> Path:
        from gtts import gTTS
        out = self._cache_path("gtts", text, "mp3")
        if not out.is_file():
            gTTS(text=text, lang=self.gtts_lang).save(str(out))
        return out

    def _vieneu_tts(self, text: str) -> Path:
        out = self._cache_path("vieneu", text, "wav")
        if not out.is_file():
            if self._vieneu is None:
                from vieneu import Vieneu
                self._vieneu = Vieneu()
            kwargs = {"voice": self.vieneu_voice} if self.vieneu_voice else {}
            audio = self._vieneu.infer(text, **kwargs)
            self._vieneu.save(audio, str(out))
        return out

    def synthesize(self, text: str) -> Path:
        text = (text or "").strip()
        if not text:
            raise TTSError("Văn bản rỗng")
        errors = []
        engines = {"edge": self._edge, "gtts": self._gtts, "vieneu": self._vieneu_tts}
        for name in self.engine_order:
            fn = engines.get(name)
            if fn is None:
                continue
            try:
                path = fn(text)
                if path.is_file() and path.stat().st_size > 0:
                    self.last_audio = path
                    log.info("TTS [%s]: \"%s\" → %s", name, text, path.name)
                    return path
                raise TTSError("file âm thanh rỗng")
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                log.warning("TTS engine %s lỗi: %s", name, exc)
                bad = self._cache_path(name, text, "wav" if name == "vieneu" else "mp3")
                if bad.exists() and bad.stat().st_size == 0:
                    bad.unlink()
        raise TTSError("Không engine TTS nào hoạt động. " + " | ".join(errors) +
                       "\nGợi ý: kiểm tra Internet (edge/gtts) hoặc cài offline: pip install vieneu")

    # ------------------------------------------------------------------ phát âm thanh
    def play(self, path: str | Path | None = None, block: bool = False) -> None:
        path = Path(path) if path else self.last_audio
        if path is None or not path.is_file():
            raise TTSError("Chưa có âm thanh để phát")
        with self._lock:
            try:
                import pygame
                if not self._mixer_ready:
                    pygame.mixer.init()
                    self._mixer_ready = True
                pygame.mixer.music.load(str(path))
                pygame.mixer.music.play()
                if block:
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(50)
                return
            except ImportError:
                pass
            except Exception as exc:
                log.warning("pygame không phát được (%s), dùng trình phát hệ thống.", exc)
        self._play_system(path, block)

    @staticmethod
    def _play_system(path: Path, block: bool) -> None:
        system = platform.system()
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        cmd = None
        if system == "Darwin":
            cmd = ["afplay", str(path)]
        else:
            for player in (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"], ["mpg123", "-q"], ["paplay"]):
                if shutil.which(player[0]):
                    cmd = player + [str(path)]
                    break
        if cmd is None:
            raise TTSError(f"Không tìm thấy trình phát âm thanh. File: {path}")
        (subprocess.run if block else subprocess.Popen)(cmd)  # type: ignore[operator]

    def stop(self) -> None:
        try:
            import pygame
            if self._mixer_ready:
                pygame.mixer.music.stop()
        except Exception:
            pass

    def speak(self, text: str, block: bool = False) -> Path:
        path = self.synthesize(text)
        self.play(path, block)
        return path

    def replay(self, block: bool = False) -> None:
        self.play(self.last_audio, block)
