from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def prepare_hf_env() -> None:
    """Whisper 权重默认从 huggingface.co 拉取。

    不要默认改 HF_ENDPOINT：部分镜像会 308 回官方站，导致 huggingface_hub
    校验失败。网络不通时再手动：export HF_ENDPOINT=https://hf-mirror.com
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass
class Settings:
    videos_dir: Path = ROOT / "videos"
    data_dir: Path = ROOT / "data"
    labels_path: Path | None = None
    frame_interval: float = 2.0
    asr_model: str = "medium"
    asr_language: str = "zh"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    ocr_lang: str = "ch"
    min_ocr_score: float = 0.5
    max_frame_width: int = 1280
    save_frames: bool = False
    skip_asr: bool = False
    skip_ocr: bool = False
    force: bool = False
    limit: int = 0

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def asr_dir(self) -> Path:
        return self.data_dir / "asr"

    @property
    def ocr_dir(self) -> Path:
        return self.data_dir / "ocr"

    @property
    def merged_dir(self) -> Path:
        return self.data_dir / "merged"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.audio_dir,
            self.asr_dir,
            self.ocr_dir,
            self.merged_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
