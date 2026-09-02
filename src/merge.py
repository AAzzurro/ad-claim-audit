from __future__ import annotations

import re
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\[\]【】()（）<>《》""\"'`]+")


def normalize_key(text: str) -> str:
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub("", text)
    return text.strip().lower()


def merge_ocr_frames(
    frames: list[dict[str, Any]],
    min_len: int = 2,
) -> list[dict[str, Any]]:
    """跨帧去重：同一句字幕只保留首次出现的时间戳。"""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for frame in frames:
        ts = frame.get("time")
        for item in frame.get("texts") or []:
            text = str(item.get("text") or "").strip()
            if len(text) < min_len:
                continue
            key = normalize_key(text)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "text": text,
                    "score": item.get("score"),
                    "time": ts,
                    "box": item.get("box"),
                }
            )
    return merged


def merge_record(
    sample_id: str,
    video_name: str,
    duration: float,
    asr: dict[str, Any] | None,
    ocr: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    asr = asr or {}
    ocr = ocr or {}
    asr_text = str(asr.get("text") or "").strip()
    ocr_lines = ocr.get("merged") or []
    ocr_text = "\n".join(str(x.get("text") or "") for x in ocr_lines).strip()
    parts = []
    if asr_text:
        parts.append(f"[ASR口播]\n{asr_text}")
    if ocr_text:
        parts.append(f"[OCR画面文字]\n{ocr_text}")
    combined = "\n\n".join(parts)
    return {
        "sample_id": sample_id,
        "video_name": video_name,
        "duration_sec": duration,
        "meta": meta or {},
        "asr_text": asr_text,
        "ocr_text": ocr_text,
        "combined_text": combined,
        "asr_segments": asr.get("segments") or [],
        "ocr_merged": ocr_lines,
    }
