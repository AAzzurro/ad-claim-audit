"""短视频虚假宣传审核台。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analyze import MODES, analyze_video, engines_loaded, ollama_status
from src.config import ROOT, Settings
from src.llm import DEFAULT_HOST, DEFAULT_MODEL, LLMError
from src.media import VIDEO_EXTS, list_videos, probe_duration
from src.review_frames import render_review_frames
from src.rules import RISK_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

WEB_DIR = ROOT / "web"
UPLOAD_DIR = ROOT / "data" / "uploads"
REVIEW_DIR = ROOT / "data" / "review_frames"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="审言 · 短视频虚假宣传检测", version="1.0")
if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_ANALYZE_LOCK = threading.Lock()


class AnalyzeIn(BaseModel):
    mode: str = "rules"
    force: bool = False
    model: str = Field(default=DEFAULT_MODEL)
    host: str = Field(default=DEFAULT_HOST)


def _settings() -> Settings:
    s = Settings(videos_dir=ROOT / "videos", data_dir=ROOT / "data")
    s.ensure_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    return s


def _put_job(job: dict[str, Any]) -> dict[str, Any]:
    with _jobs_lock:
        _jobs[job["id"]] = job
    return job


def _get_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "任务不存在或已过期，请重新选择视频")
    return job


def _safe_library_path(sample_id: str) -> Path:
    videos = list_videos(ROOT / "videos")
    for path in videos:
        if path.stem == sample_id or path.name == sample_id:
            return path
    raise HTTPException(404, f"测试集中没有 {sample_id}")


def _job_public(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["id"],
        "sample_id": job["sample_id"],
        "video_name": job["video_name"],
        "duration_sec": job.get("duration_sec"),
        "from_library": job["from_library"],
        "media_url": f"/api/jobs/{job['id']}/media",
    }


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "缺少 web/index.html")
    return FileResponse(page)


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    ollama = ollama_status()
    videos = list_videos(ROOT / "videos")
    return {
        "modes": list(MODES.values()),
        "risk_labels": list(RISK_LABELS),
        "ollama": ollama,
        "engines_loaded": engines_loaded(),
        "library_count": len(videos),
        "llm_model": DEFAULT_MODEL,
    }


@app.get("/api/library")
def library() -> dict[str, Any]:
    settings = _settings()
    items = []
    for path in list_videos(ROOT / "videos"):
        sid = path.stem
        items.append(
            {
                "id": sid,
                "name": path.name,
                "cached": (settings.asr_dir / f"{sid}.json").exists()
                and (settings.ocr_dir / f"{sid}.json").exists(),
            }
        )
    return {"items": items, "n": len(items)}


@app.post("/api/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    library_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    if library_id and str(library_id).strip():
        path = _safe_library_path(str(library_id).strip())
        job = {
            "id": uuid.uuid4().hex[:12],
            "sample_id": path.stem,
            "video_path": str(path),
            "video_name": path.name,
            "duration_sec": round(probe_duration(path), 3),
            "from_library": True,
        }
        _put_job(job)
        return _job_public(job)

    if file is None or not file.filename:
        raise HTTPException(400, "请上传视频，或选择测试集中的一条")
    suffix = Path(file.filename).suffix
    if suffix not in VIDEO_EXTS:
        raise HTTPException(400, f"不支持的视频格式 {suffix}，请使用 mp4 / mov / mkv / webm")

    job_id = uuid.uuid4().hex[:12]
    dest_dir = UPLOAD_DIR / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"input{suffix.lower()}"
    size = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "视频超过 200MB")
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "上传文件为空")

    job = {
        "id": job_id,
        "sample_id": f"u_{job_id}",
        "video_path": str(dest),
        "video_name": Path(file.filename).name,
        "duration_sec": round(probe_duration(dest), 3),
        "from_library": False,
    }
    _put_job(job)
    return _job_public(job)


@app.get("/api/jobs/{job_id}/media")
def job_media(job_id: str) -> FileResponse:
    job = _get_job(job_id)
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(404, "视频文件已丢失")
    return FileResponse(path, media_type="video/mp4", filename=job["video_name"])


@app.get("/api/jobs/{job_id}/frames/{name}")
def job_frame(job_id: str, name: str) -> FileResponse:
    job = _get_job(job_id)
    frames_dir = Path(job.get("frames_dir") or "")
    safe = Path(name).name
    if not frames_dir or safe != name or not safe.endswith(".jpg"):
        raise HTTPException(404, "画面不存在")
    path = frames_dir / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "画面不存在")
    return FileResponse(path, media_type="image/jpeg", filename=safe)


def _attach_problem_frames(job: dict[str, Any], result: dict[str, Any], settings: Settings) -> dict[str, Any]:
    if result.get("verdict") != "risk":
        result["problem_frames"] = []
        return result
    dest = REVIEW_DIR / job["id"]
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.jpg"):
        old.unlink(missing_ok=True)
    ocr_path = settings.ocr_dir / f"{job['sample_id']}.json"
    ocr = {}
    if ocr_path.exists():
        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    frames = render_review_frames(
        Path(job["video_path"]),
        ocr,
        list(result.get("evidence") or []),
        dest,
        max_width=settings.max_frame_width,
    )
    job["frames_dir"] = str(dest)
    _put_job(job)
    result["problem_frames"] = [
        {
            **item,
            "url": f"/api/jobs/{job['id']}/frames/{item['image_name']}",
        }
        for item in frames
    ]
    return result


@app.post("/api/jobs/{job_id}/analyze")
async def analyze_job(job_id: str, body: Optional[AnalyzeIn] = None) -> StreamingResponse:
    job = _get_job(job_id)
    payload = body or AnalyzeIn()
    mode = payload.mode.strip()
    if mode not in MODES:
        raise HTTPException(400, f"未知模式 {mode}")
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress(stage: str, message: str, extra: dict[str, Any] | None = None) -> None:
        event = {"stage": stage, "message": message, **(extra or {})}
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def work() -> None:
        try:
            if not _ANALYZE_LOCK.acquire(blocking=False):
                progress("start", "等待上一条审核结束…")
                _ANALYZE_LOCK.acquire()
            try:
                progress("start", "开始审核")
                settings = _settings()
                result = analyze_video(
                    Path(job["video_path"]),
                    job["sample_id"],
                    mode,
                    force=payload.force,
                    on_progress=progress,
                    settings=settings,
                    llm_model=payload.model,
                    llm_host=payload.host,
                )
                if result.get("verdict") == "risk":
                    progress("review", "正在截取问题画面…")
                    result = _attach_problem_frames(job, result, settings)
                else:
                    result["problem_frames"] = []
                result["job_id"] = job_id
                result["from_library"] = job["from_library"]
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"stage": "done", "message": "审核完成", "result": result}
                )
            finally:
                _ANALYZE_LOCK.release()
        except LLMError as exc:
            logger.warning("模型不可用: %s", exc)
            loop.call_soon_threadsafe(
                queue.put_nowait, {"stage": "error", "message": str(exc)}
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("审核失败")
            loop.call_soon_threadsafe(
                queue.put_nowait, {"stage": "error", "message": f"审核失败：{exc}"}
            )

    threading.Thread(target=work, daemon=True, name=f"analyze-{job_id}").start()

    async def events():
        while True:
            item = await queue.get()
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            if item.get("stage") in {"done", "error"}:
                break

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动审言审核台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    import uvicorn

    logger.info("审言审核台 http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
