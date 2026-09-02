from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.asr import build_asr_engine, save_json
from src.config import ROOT, MODELS_DIR, Settings, prepare_hf_env
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
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="本地权重目录，默认 <repo>/models")
    parser.add_argument("--labels", type=Path, default=None, help="标签表 csv/xlsx，可稍后补上")
    parser.add_argument("--interval", type=float, default=3.0, help="抽帧间隔（秒），任务要求 1–3 秒")
    parser.add_argument("--frame-change-threshold", type=float, default=0.015, help="相邻帧变化阈值，低于此值跳过 OCR；0 表示关闭")
    parser.add_argument(
        "--asr-backend",
        default="sensevoice",
        choices=["whisper", "sensevoice"],
        help="ASR 后端：sensevoice（默认，中文更准）或 whisper",
    )
    parser.add_argument(
        "--asr-model",
        default="small",
        help="faster-whisper 尺寸（tiny/base/small/medium/large-v3）或本地模型目录",
    )
    parser.add_argument(
        "--ocr-size",
        default="small",
        choices=["tiny", "small", "medium"],
        help="PP-OCRv6 档位，默认 small；medium 更准更慢",
    )
    parser.add_argument("--ocr-det-dir", type=Path, default=None, help="本地 OCR 检测模型目录")
    parser.add_argument("--ocr-rec-dir", type=Path, default=None, help="本地 OCR 识别模型目录")
    parser.add_argument("--only", default="", help="只处理这些样本，逗号分隔，如 1,10 或 1.mp4,10.mp4")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条，0 表示全部")
    parser.add_argument("--save-frames", action="store_true", help="保留抽帧图片到 data/frames/")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，重新抽取")
    args = parser.parse_args(argv)
    return Settings(
        videos_dir=args.videos_dir,
        data_dir=args.out_dir,
        models_dir=args.models_dir,
        labels_path=args.labels,
        frame_interval=args.interval,
        frame_change_threshold=max(0.0, args.frame_change_threshold),
        asr_backend=args.asr_backend,
        asr_model=args.asr_model,
        ocr_size=args.ocr_size,
        ocr_det_dir=args.ocr_det_dir,
        ocr_rec_dir=args.ocr_rec_dir,
        only=args.only,
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
    asr_engine: Any | None,
    ocr_engine: PaddleOCREngine | None,
    meta: dict[str, Any],
) -> dict[str, Any]:
    sample_id = _pick_sample_id(meta, video_path.stem)
    duration = probe_duration(video_path)
    asr_path = settings.asr_dir / f"{sample_id}.json"
    ocr_path = settings.ocr_dir / f"{sample_id}.json"
    merged_path = settings.merged_dir / f"{sample_id}.json"
    wav_path = settings.audio_dir / f"{sample_id}.wav"

    # --force 只重跑未 skip 的模态；skip-ocr 时仍读已有 frames 做规则重洗。
    asr_payload = None if (settings.force and not settings.skip_asr) else _load_json(asr_path)
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

    ocr_payload = None if (settings.force and not settings.skip_ocr) else _load_json(ocr_path)
    if ocr_payload is not None and ocr_payload.get("frames"):
        # 让升级后的清洗规则也能作用于历史 OCR 中间结果，无需重新推理。
        ocr_payload["merged"] = merge_ocr_frames(ocr_payload["frames"])
    if not settings.skip_ocr and ocr_payload is None:
        if ocr_engine is None:
            raise RuntimeError("OCR 引擎未初始化")
        logger.info("OCR %s interval=%.1fs", video_path.name, settings.frame_interval)
        frame_records: list[dict[str, Any]] = []
        frames = iter_frames(video_path, settings.frame_interval, settings.max_frame_width)
        for idx, (ts, image) in enumerate(frames):
            # 直播静止画面连续出现时只识别一次，减少 PaddleOCR 调用次数。
            if frame_records and settings.frame_change_threshold > 0:
                import cv2
                import numpy as np
                previous = frames[idx - 1][1]
                a = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (96, 96))
                b = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (96, 96))
                change = float(np.mean(cv2.absdiff(a, b))) / 255.0
                if change < settings.frame_change_threshold:
                    continue
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
    if settings.only:
        wanted = {item.strip() for item in settings.only.split(",") if item.strip()}
        keys = set()
        for item in wanted:
            keys.add(item)
            keys.add(Path(item).name)
            keys.add(Path(item).stem)
        videos = [path for path in videos if path.name in keys or path.stem in keys]
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

    asr_engine = None if settings.skip_asr else build_asr_engine(
        backend=settings.asr_backend,
        model_size=settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        language=settings.asr_language,
        models_dir=settings.models_dir,
    )
    ocr_engine = None if settings.skip_ocr else PaddleOCREngine(
        lang=settings.ocr_lang,
        min_score=settings.min_ocr_score,
        models_dir=settings.models_dir,
        det_dir=settings.ocr_det_dir,
        rec_dir=settings.ocr_rec_dir,
        ocr_size=settings.ocr_size,
    )

    index_rows = []
    failures = []
    for video_path in tqdm(videos, desc="抽取 ASR/OCR"):
        meta = lookup_meta(label_index, video_path)
        try:
            merged = process_one(video_path, settings, asr_engine, ocr_engine, meta)
            sample_id = _pick_sample_id(meta, video_path.stem)
            index_rows.append(
                {
                    "sample_id": sample_id,
                    "video_name": video_path.name,
                    "duration_sec": probe_duration(video_path),
                    "asr_chars": len(merged.get("audio_text") or ""),
                    "ocr_chars": len(merged.get("visual_text") or ""),
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
                    "ocr_chars": 0,
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
            "frame_change_threshold": settings.frame_change_threshold,
            "asr_backend": settings.asr_backend,
            "asr_model": settings.asr_model,
            "ocr_size": settings.ocr_size,
            "models_dir": str(settings.models_dir),
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
