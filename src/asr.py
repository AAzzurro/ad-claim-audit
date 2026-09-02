from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import prepare_hf_env

logger = logging.getLogger(__name__)

_MODELSCOPE_REPOS = {
    "tiny": "pengzhendong/faster-whisper-tiny",
    "base": "pengzhendong/faster-whisper-base",
    "small": "pengzhendong/faster-whisper-small",
    "medium": "pengzhendong/faster-whisper-medium",
    "large-v3": "pengzhendong/faster-whisper-large-v3",
    "large": "pengzhendong/faster-whisper-large-v3",
}


def _build_whisper_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HuggingFace 加载失败（%s），改从 ModelScope 下载", exc)
        repo = _MODELSCOPE_REPOS.get(model_size, f"pengzhendong/faster-whisper-{model_size}")
        from modelscope.hub.snapshot_download import snapshot_download

        local_dir = snapshot_download(repo)
        return WhisperModel(local_dir, device=device, compute_type=compute_type)


class WhisperASR:
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
    ) -> None:
        prepare_hf_env()
        logger.info("加载 Whisper 模型 %s (%s/%s)", model_size, device, compute_type)
        self.language = language
        self.model = _build_whisper_model(model_size, device, compute_type)

    def transcribe(self, audio_path: Path) -> dict[str, Any]:
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            beam_size=5,
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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
