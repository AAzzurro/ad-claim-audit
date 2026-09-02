from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.config import MODELS_DIR
from src.local_models import ensure_whisper_dir

logger = logging.getLogger(__name__)

_SENSEVOICE_TAG_RE = re.compile(r"<\|[^>]+\|>")
_SENSEVOICE_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+"
)


def _build_whisper_model(
    model_size: str,
    device: str,
    compute_type: str,
    models_dir: Path | None = None,
):
    from faster_whisper import WhisperModel

    local_dir = ensure_whisper_dir(model_size, models_dir or MODELS_DIR)
    return WhisperModel(str(local_dir), device=device, compute_type=compute_type)


class WhisperASR:
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
        models_dir: Path | None = None,
    ) -> None:
        logger.info("加载 Whisper 模型 %s (%s/%s)", model_size, device, compute_type)
        self.language = language
        self.model = _build_whisper_model(model_size, device, compute_type, models_dir)

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # beam=1 在 CPU 上明显更快；广告口播通常不需要宽 beam 搜索。
            beam_size=1,
            best_of=1,
            temperature=0.0,
            word_timestamps=False,
        )
        segments = []
        texts = []
        for seg in segments_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            item = {
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
            }
            segments.append(item)
            texts.append(text)
        return {
            "language": getattr(info, "language", self.language),
            "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
            "text": " ".join(texts).strip(),
            "segments": segments,
        }


def _sensevoice_language(language: str) -> str:
    mapping = {
        "zh": "zn",
        "cn": "zn",
        "chinese": "zn",
        "yue": "yue",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "auto": "auto",
    }
    return mapping.get((language or "zh").strip().lower(), "auto")


def _strip_sensevoice(text: str) -> str:
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        text = str(rich_transcription_postprocess(text) or "").strip()
        text = _SENSEVOICE_EMOJI_RE.sub(" ", text)
        return " ".join(text.split()).strip()
    except Exception:  # noqa: BLE001
        cleaned = _SENSEVOICE_TAG_RE.sub(" ", text)
        cleaned = _SENSEVOICE_EMOJI_RE.sub(" ", cleaned)
        return " ".join(cleaned.split()).strip()


def _pick_torch_device(preferred: str | None = None) -> str:
    if preferred and preferred not in {"auto", "cpu"}:
        return preferred
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class SenseVoiceASR:
    def __init__(
        self,
        language: str = "zh",
        models_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        from funasr import AutoModel

        from src.local_models import ensure_sensevoice_dir

        local_dir = ensure_sensevoice_dir(models_dir or MODELS_DIR)
        self.language = language
        self.device = _pick_torch_device(device)
        logger.info("加载 SenseVoiceSmall (%s) <- %s", self.device, local_dir)
        try:
            self.model = AutoModel(
                model=str(local_dir),
                device=self.device,
                disable_update=True,
            )
        except Exception as exc:  # noqa: BLE001
            if self.device == "cpu":
                raise
            logger.warning("SenseVoice 在 %s 上加载失败（%s），改用 CPU", self.device, exc)
            self.device = "cpu"
            self.model = AutoModel(
                model=str(local_dir),
                device="cpu",
                disable_update=True,
            )

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        raw = self.model.generate(
            input=str(audio_path),
            language=_sensevoice_language(self.language),
            use_itn=True,
            batch_size_s=60,
        )
        item = raw[0] if raw else {}
        text = _strip_sensevoice(str(item.get("text") or ""))
        duration = float(item.get("duration") or 0.0)
        timestamps = item.get("timestamp") or []
        segments = []
        if timestamps and text:
            segments = [
                {
                    "start": round(float(ts[0]) / (1000.0 if ts[0] > 100 else 1.0), 3),
                    "end": round(float(ts[1]) / (1000.0 if ts[1] > 100 else 1.0), 3),
                    "text": text,
                }
                for ts in timestamps
                if isinstance(ts, (list, tuple)) and len(ts) >= 2
            ]
        if not segments:
            segments = [{"start": 0.0, "end": round(duration, 3), "text": text}] if text else []
        return {
            "language": self.language,
            "duration": round(duration, 3),
            "text": text,
            "segments": segments,
        }


def build_asr_engine(
    backend: str = "whisper",
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str = "zh",
    models_dir: Path | None = None,
):
    name = (backend or "whisper").strip().lower()
    if name in {"sensevoice", "funasr"}:
        return SenseVoiceASR(language=language, models_dir=models_dir, device=device)
    if name in {"whisper", "faster-whisper", "faster_whisper"}:
        return WhisperASR(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            models_dir=models_dir,
        )
    raise ValueError(f"未知 ASR 后端 {backend!r}，可选 whisper / sensevoice")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
