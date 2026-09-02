from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.asr import WhisperASR, save_json
from src.config import ROOT, Settings, prepare_hf_env
from src.labels import discover_labels, index_labels, load_labels, lookup_meta
from src.media import extract_audio, iter_frames, list_videos, probe_duration
from src.merge import merge_ocr_frames, merge_record
from src.ocr import PaddleOCREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(
        description="对 videos/ 中的短视频做音频提取、中文 ASR 和画面 OCR。",
    )
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "videos")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--labels", type=Path, default=None, help="标签表 csv/xlsx，可稍后补上")
    parser.add_argument("--interval", type=float, default=2.0, help="抽帧间隔（秒），任务要求 1–3 秒")
    parser.add_argument("--asr-model", default="medium", help="faster-whisper 模型：tiny/base/small/medium/large-v3")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条，0 表示全部")
    parser.add_argument("--save-frames", action="store_true", help="保留抽帧图片到 data/frames/")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，重新抽取")
    args = parser.parse_args(argv)
    return Settings(
        videos_dir=args.videos_dir,
        data_dir=args.out_dir,
        labels_path=args.labels,
        frame_interval=args.interval,
        asr_model=args.asr_model,
        save_frames=args.save_frames,
        skip_asr=args.skip_asr,
        skip_ocr=args.skip_ocr,
        force=args.force,
        limit=args.limit,
    )


def _pick_sample_id(meta: dict[str, Any], fallback: str) -> str:
    for key in ("sample_id", "样本编号", "编号", "id", "ID"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return fallback


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame(path: Path, image) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def process_one(
    video_path: Path,
    settings: Settings,
    asr_engine: WhisperASR | None,
    ocr_engine: PaddleOCREngine | None,
    meta: dict[str, Any],
) -> dict[str, Any]:
    sample_id = _pick_sample_id(meta, video_path.stem)
    duration = probe_duration(video_path)
    asr_path = settings.asr_dir / f"{sample_id}.json"
    ocr_path = settings.ocr_dir / f"{sample_id}.json"
    merged_path = settings.merged_dir / f"{sample_id}.json"
    wav_path = settings.audio_dir / f"{sample_id}.wav"

    asr_payload = None if settings.force else _load_json(asr_path)
    if not settings.skip_asr and asr_payload is None:
        if asr_engine is None:
            raise RuntimeError("ASR 引擎未初始化")
        logger.info("ASR %s", video_path.name)
        extract_audio(video_path, wav_path)
        asr_payload = asr_engine.transcribe(wav_path)
        asr_payload.update(
            {
                "sample_id": sample_id,
                "video_name": video_path.name,
                "audio_path": str(wav_path.relative_to(settings.data_dir)),
            }
        )
        save_json(asr_path, asr_payload)
    elif settings.skip_asr:
        asr_payload = asr_payload or {"text": "", "segments": []}

    ocr_payload = None if settings.force else _load_json(ocr_path)
    if not settings.skip_ocr and ocr_payload is None:
        if ocr_engine is None:
            raise RuntimeError("OCR 引擎未初始化")
        logger.info("OCR %s interval=%.1fs", video_path.name, settings.frame_interval)
        frame_records: list[dict[str, Any]] = []
        frames = iter_frames(video_path, settings.frame_interval, settings.max_frame_width)
        for idx, (ts, image) in enumerate(frames):
            texts = ocr_engine.recognize_image(image)
            if settings.save_frames:
                frame_file = settings.frames_dir / sample_id / f"{idx:04d}_{ts:.2f}s.jpg"
                _write_frame(frame_file, image)
            frame_records.append({"time": ts, "texts": texts, "n_texts": len(texts)})
        ocr_payload = {
            "sample_id": sample_id,
            "video_name": video_path.name,
            "interval_sec": settings.frame_interval,
            "n_frames": len(frame_records),
            "frames": frame_records,
            "merged": merge_ocr_frames(frame_records),
        }
        save_json(ocr_path, ocr_payload)
    elif settings.skip_ocr:
        ocr_payload = ocr_payload or {"frames": [], "merged": []}

    merged = merge_record(
        sample_id=sample_id,
        video_name=video_path.name,
        duration=duration,
        asr=asr_payload,
        ocr=ocr_payload,
        meta=meta,
    )
    save_json(merged_path, merged)
    return merged


def run(settings: Settings) -> int:
    prepare_hf_env()
    settings.ensure_dirs()
    videos = list_videos(settings.videos_dir)
    if settings.limit:
        videos = videos[: settings.limit]

    if not videos:
        logger.warning(
            "未在 %s 找到视频。把 mp4 放到该目录后执行: python -m src.pipeline",
            settings.videos_dir,
        )
        return 0

    labels_path = settings.labels_path
    if labels_path is None:
        labels_path = discover_labels([ROOT, settings.videos_dir, settings.data_dir, ROOT / "data" / "labels"])
    label_index: dict[str, dict[str, Any]] = {}
    if labels_path and labels_path.exists():
        logger.info("读取标签表 %s", labels_path)
        label_index = index_labels(load_labels(labels_path))
    else:
        logger.info("暂未发现标签表，sample_id 将使用视频文件名（不含后缀）")

    asr_engine = None if settings.skip_asr else WhisperASR(
        model_size=settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        language=settings.asr_language,
    )
    ocr_engine = None if settings.skip_ocr else PaddleOCREngine(
        lang=settings.ocr_lang,
        min_score=settings.min_ocr_score,
    )

    index_rows = []
    failures = []
    for video_path in tqdm(videos, desc="抽取 ASR/OCR"):
        meta = lookup_meta(label_index, video_path)
        try:
            merged = process_one(video_path, settings, asr_engine, ocr_engine, meta)
            index_rows.append(
                {
                    "sample_id": merged["sample_id"],
                    "video_name": merged["video_name"],
                    "duration_sec": merged["duration_sec"],
                    "asr_chars": len(merged.get("asr_text") or ""),
                    "ocr_lines": len(merged.get("ocr_merged") or []),
                    "ok": True,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("处理失败 %s", video_path.name)
            failures.append(video_path.name)
            index_rows.append(
                {
                    "sample_id": video_path.stem,
                    "video_name": video_path.name,
                    "duration_sec": None,
                    "asr_chars": 0,
                    "ocr_lines": 0,
                    "ok": False,
                    "error": str(exc),
                }
            )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_videos": len(videos),
        "n_ok": sum(1 for r in index_rows if r["ok"]),
        "n_failed": len(failures),
        "failures": failures,
        "settings": {
            "frame_interval": settings.frame_interval,
            "asr_model": settings.asr_model,
            "skip_asr": settings.skip_asr,
            "skip_ocr": settings.skip_ocr,
        },
        "items": index_rows,
    }
    save_json(settings.data_dir / "extract_index.json", summary)
    logger.info(
        "完成 %s/%s，失败 %s。索引: %s",
        summary["n_ok"],
        summary["n_videos"],
        summary["n_failed"],
        settings.data_dir / "extract_index.json",
    )
    return 1 if failures else 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
