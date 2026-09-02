from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.asr import build_asr_engine, save_json
from src.config import MODELS_DIR, ROOT, Settings
from src.media import extract_audio, iter_frames, probe_duration
from src.merge import merge_ocr_frames, merge_record
from src.ocr import PaddleOCREngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare")


def _ocr_one(
    video_path: Path,
    ocr_engine: PaddleOCREngine,
    settings: Settings,
) -> tuple[dict[str, Any], float, int]:
    frames = iter_frames(video_path, settings.frame_interval, settings.max_frame_width)
    records: list[dict[str, Any]] = []
    skipped = 0
    started = time.perf_counter()
    for idx, (ts, image) in enumerate(frames):
        if records and settings.frame_change_threshold > 0:
            import cv2
            import numpy as np

            previous = frames[idx - 1][1]
            a = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (96, 96))
            b = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (96, 96))
            change = float(np.mean(cv2.absdiff(a, b))) / 255.0
            if change < settings.frame_change_threshold:
                skipped += 1
                continue
        texts = ocr_engine.recognize_image(image)
        records.append({"time": ts, "texts": texts, "n_texts": len(texts)})
    elapsed = time.perf_counter() - started
    payload = {
        "sample_id": video_path.stem,
        "video_name": video_path.name,
        "n_frames": len(records),
        "n_skipped": skipped,
        "frames": records,
        "merged": merge_ocr_frames(records),
    }
    return payload, elapsed, skipped


def _run_asr(name: str, factory, videos: list[Path], wavs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    logger.info("加载 ASR：%s", name)
    load_t0 = time.perf_counter()
    engine = factory()
    load_s = time.perf_counter() - load_t0
    items = []
    for video in videos:
        wav = wavs[video.stem]
        infer_t0 = time.perf_counter()
        payload = engine.transcribe(wav)
        infer_s = time.perf_counter() - infer_t0
        payload.update({"sample_id": video.stem, "video_name": video.name})
        save_json(out_dir / "asr" / f"{video.stem}.json", payload)
        items.append(
            {
                "sample_id": video.stem,
                "seconds": round(infer_s, 3),
                "chars": len(payload.get("text") or ""),
                "text": payload.get("text") or "",
            }
        )
        logger.info("ASR %s %s %.2fs  %s 字", name, video.name, infer_s, items[-1]["chars"])
    del engine
    gc.collect()
    return {"backend": name, "load_seconds": round(load_s, 3), "items": items}


def _run_ocr(size: str, videos: list[Path], settings: Settings, out_dir: Path) -> dict[str, Any]:
    logger.info("加载 OCR：PP-OCRv6 %s", size)
    load_t0 = time.perf_counter()
    engine = PaddleOCREngine(ocr_size=size, models_dir=settings.models_dir)
    load_s = time.perf_counter() - load_t0
    items = []
    for video in videos:
        payload, infer_s, skipped = _ocr_one(video, engine, settings)
        save_json(out_dir / "ocr" / f"{video.stem}.json", payload)
        merged = merge_record(
            sample_id=video.stem,
            video_name=video.name,
            duration=probe_duration(video),
            asr={"text": ""},
            ocr=payload,
        )
        items.append(
            {
                "sample_id": video.stem,
                "seconds": round(infer_s, 3),
                "n_frames": payload["n_frames"],
                "n_skipped": skipped,
                "chars": len(merged.get("visual_text") or ""),
                "text": merged.get("visual_text") or "",
            }
        )
        logger.info("OCR %s %s %.2fs  %s 字", size, video.name, infer_s, items[-1]["chars"])
    del engine
    gc.collect()
    return {"backend": f"ppocrv6-{size}", "load_seconds": round(load_s, 3), "items": items}


def main() -> int:
    parser = argparse.ArgumentParser(description="对比 Whisper/SenseVoice 与 PP-OCRv6 medium/small")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "videos")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "compare")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--only", default="1,10")
    args = parser.parse_args()

    settings = Settings(
        videos_dir=args.videos_dir,
        data_dir=args.out_dir,
        models_dir=args.models_dir,
        only=args.only,
    )
    settings.ensure_dirs()
    (args.out_dir / "asr").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ocr").mkdir(parents=True, exist_ok=True)

    wanted = {item.strip() for item in args.only.split(",") if item.strip()}
    videos = []
    for item in wanted:
        for candidate in (args.videos_dir / item, args.videos_dir / f"{item}.mp4"):
            if candidate.exists():
                videos.append(candidate)
                break
    videos = sorted(videos, key=lambda p: p.name)
    if not videos:
        logger.error("未找到对比视频：%s", args.only)
        return 1

    wavs: dict[str, Path] = {}
    audio_dir = ROOT / "data" / "audio"
    for video in videos:
        wav = audio_dir / f"{video.stem}.wav"
        if not wav.exists():
            logger.info("抽音频 %s", video.name)
            extract_audio(video, wav)
        wavs[video.stem] = wav

    from src.local_models import ensure_ocr_dirs, ensure_sensevoice_dir, ensure_whisper_dir

    logger.info("预下载 / 校验本地权重")
    ensure_whisper_dir("small", settings.models_dir)
    ensure_sensevoice_dir(settings.models_dir)
    ensure_ocr_dirs(settings.models_dir, size="medium")
    ensure_ocr_dirs(settings.models_dir, size="small")

    report: dict[str, Any] = {"videos": [v.name for v in videos], "asr": [], "ocr": []}

    report["asr"].append(
        _run_asr(
            "whisper-small",
            lambda: build_asr_engine("whisper", model_size="small", models_dir=settings.models_dir),
            videos,
            wavs,
            args.out_dir / "whisper",
        )
    )
    report["asr"].append(
        _run_asr(
            "sensevoice-small",
            lambda: build_asr_engine("sensevoice", models_dir=settings.models_dir),
            videos,
            wavs,
            args.out_dir / "sensevoice",
        )
    )
    report["ocr"].append(_run_ocr("medium", videos, settings, args.out_dir / "ocr-medium"))
    report["ocr"].append(_run_ocr("small", videos, settings, args.out_dir / "ocr-small"))

    for video in videos:
        whisper = next(x for x in report["asr"][0]["items"] if x["sample_id"] == video.stem)
        sense = next(x for x in report["asr"][1]["items"] if x["sample_id"] == video.stem)
        medium = next(x for x in report["ocr"][0]["items"] if x["sample_id"] == video.stem)
        small = next(x for x in report["ocr"][1]["items"] if x["sample_id"] == video.stem)
        save_json(
            args.out_dir / "merged" / f"{video.stem}.json",
            {
                "whisper_audio_text": whisper["text"],
                "sensevoice_audio_text": sense["text"],
                "ocr_medium_visual_text": medium["text"],
                "ocr_small_visual_text": small["text"],
            },
        )

    save_json(args.out_dir / "report.json", report)
    logger.info("对比完成：%s", args.out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
