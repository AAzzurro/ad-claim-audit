from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".MP4", ".MOV"}
_VIDEO_EXTS = VIDEO_EXTS


def find_ffmpeg() -> str:
    """优先系统 ffmpeg，否则使用 imageio-ffmpeg 自带二进制。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "未找到 ffmpeg。请安装 imageio-ffmpeg，或把 ffmpeg 加入 PATH。"
        ) from exc


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", *args]
    logger.debug("ffmpeg %s", " ".join(cmd[3:]))
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def probe_duration(video_path: Path) -> float:
    ffmpeg = find_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    text = (proc.stderr or "") + (proc.stdout or "")
    match = _DURATION_RE.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps > 0 and frames > 0:
            return float(frames / fps)
    finally:
        cap.release()
    return 0.0


def list_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.exists():
        return []
    files = [
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix in _VIDEO_EXTS
    ]
    return sorted(files, key=lambda p: p.name)


def extract_audio(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> Path:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError(f"音频提取失败: {video_path}")
    return wav_path


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def iter_frames(
    video_path: Path,
    interval: float,
    max_width: int = 1280,
) -> list[tuple[float, np.ndarray]]:
    """按固定间隔抽帧。优先 OpenCV 定位，失败则回退 ffmpeg。"""
    frames = _iter_frames_cv2(video_path, interval, max_width)
    if frames:
        return frames
    logger.warning("OpenCV 抽帧为空，改用 ffmpeg: %s", video_path.name)
    return _iter_frames_ffmpeg(video_path, interval, max_width)


def _iter_frames_cv2(
    video_path: Path,
    interval: float,
    max_width: int,
) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if total else probe_duration(video_path)
    if duration <= 0:
        duration = 0.0

    out: list[tuple[float, np.ndarray]] = []
    t = 0.0
    # 至少抽第 0 秒；末尾再补一帧，避免漏掉最后的字幕
    timestamps = []
    while t <= duration + 1e-3:
        timestamps.append(round(t, 3))
        t += interval
    if duration > 0 and (not timestamps or timestamps[-1] < duration - 0.15):
        timestamps.append(round(duration, 3))

    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out.append((ts, resize_frame(frame, max_width)))
    finally:
        cap.release()
    return out


def _iter_frames_ffmpeg(
    video_path: Path,
    interval: float,
    max_width: int,
) -> list[tuple[float, np.ndarray]]:
    import tempfile

    duration = probe_duration(video_path)
    with tempfile.TemporaryDirectory(prefix="ocr_frames_") as tmp:
        tmp_dir = Path(tmp)
        pattern = tmp_dir / "f_%06d.jpg"
        vf = f"fps=1/{interval}"
        if max_width:
            vf = f"{vf},scale='min({max_width},iw)':-2"
        run_ffmpeg(["-y", "-i", str(video_path), "-vf", vf, "-q:v", "2", str(pattern)])
        files = sorted(tmp_dir.glob("f_*.jpg"))
        out: list[tuple[float, np.ndarray]] = []
        for idx, file in enumerate(files):
            image = cv2.imread(str(file))
            if image is None:
                continue
            ts = round(idx * interval, 3)
            if duration and ts > duration:
                ts = round(duration, 3)
            out.append((ts, image))
        return out
