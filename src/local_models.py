from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.config import MODELS_DIR, ROOT, prepare_hf_env

logger = logging.getLogger(__name__)

WHISPER_MARKERS = ("model.bin", "model.safetensors")
OCR_MARKERS = ("inference.yml", "inference.pdiparams")
SENSEVOICE_MARKERS = ("model.pt", "model.onnx", "model.onnx.quant")

OCR_VARIANTS = {
    "medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
    "small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "tiny": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
}
OCR_DET_NAME, OCR_REC_NAME = OCR_VARIANTS["small"]
OCR_HF_REPOS = {
    name: f"PaddlePaddle/{name}" for pair in OCR_VARIANTS.values() for name in pair
}

SENSEVOICE_DIR_NAME = "SenseVoiceSmall"
SENSEVOICE_MS_REPO = "iic/SenseVoiceSmall"
SENSEVOICE_HF_REPO = "FunAudioLLM/SenseVoiceSmall"

WHISPER_HF_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
}
WHISPER_MS_REPOS = {
    "tiny": "pengzhendong/faster-whisper-tiny",
    "base": "pengzhendong/faster-whisper-base",
    "small": "pengzhendong/faster-whisper-small",
    "medium": "pengzhendong/faster-whisper-medium",
    "large-v3": "pengzhendong/faster-whisper-large-v3",
    "large": "pengzhendong/faster-whisper-large-v3",
}
WHISPER_DIR_NAMES = {
    "tiny": "faster-whisper-tiny",
    "base": "faster-whisper-base",
    "small": "faster-whisper-small",
    "medium": "faster-whisper-medium",
    "large-v3": "faster-whisper-large-v3",
    "large": "faster-whisper-large-v3",
}


def ocr_model_names(size: str) -> tuple[str, str]:
    key = (size or "small").strip().lower()
    if key not in OCR_VARIANTS:
        raise ValueError(f"未知 OCR 档位 {size!r}，可选: {', '.join(OCR_VARIANTS)}")
    return OCR_VARIANTS[key]


def is_complete_model_dir(path: Path | None, markers: tuple[str, ...]) -> bool:
    if path is None or not path.is_dir():
        return False
    return any((path / name).is_file() for name in markers)


def looks_like_path(value: str) -> bool:
    path = Path(value).expanduser()
    return path.is_absolute() or path.exists() or "/" in value or value.startswith(".")


def resolve_user_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def copy_model_dir(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dest,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".cache", ".lock", ".locks"),
    )
    return dest


def _download_hf(repo_id: str, dest: Path) -> Path:
    prepare_hf_env()
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, local_dir=str(dest))
    return dest


def _download_modelscope(repo_id: str, dest: Path) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, local_dir=str(dest))
    return dest


def _paddlex_dir(name: str) -> Path:
    return Path.home() / ".paddlex" / "official_models" / name


def whisper_size_key(model_size: str) -> str:
    key = model_size.strip()
    if key in WHISPER_DIR_NAMES:
        return key
    return key


def default_whisper_dir(model_size: str, models_dir: Path) -> Path:
    key = whisper_size_key(model_size)
    return models_dir / WHISPER_DIR_NAMES.get(key, f"faster-whisper-{key}")


def find_local_whisper(model_size: str, models_dir: Path) -> Path | None:
    if looks_like_path(model_size):
        path = resolve_user_path(model_size)
        if is_complete_model_dir(path, WHISPER_MARKERS):
            return path
        raise FileNotFoundError(f"本地 Whisper 目录无效或不完整: {path}")

    key = whisper_size_key(model_size)
    dirname = WHISPER_DIR_NAMES.get(key, f"faster-whisper-{key}")
    candidates = [
        models_dir / dirname,
        models_dir / f"whisper-{key}",
        models_dir / key,
        models_dir / model_size,
    ]
    for path in candidates:
        if is_complete_model_dir(path, WHISPER_MARKERS):
            return path
    return None


def ensure_whisper_dir(model_size: str, models_dir: Path | None = None) -> Path:
    models_dir = models_dir or MODELS_DIR
    local = find_local_whisper(model_size, models_dir)
    if local is not None:
        logger.info("使用本地 Whisper: %s", local)
        return local

    dest = default_whisper_dir(model_size, models_dir)
    key = whisper_size_key(model_size)
    hf_repo = WHISPER_HF_REPOS.get(key, f"Systran/faster-whisper-{key}")
    ms_repo = WHISPER_MS_REPOS.get(key, f"pengzhendong/faster-whisper-{key}")

    try:
        logger.info("本地无 Whisper，从 HuggingFace 下载到 %s", dest)
        _download_hf(hf_repo, dest)
        if is_complete_model_dir(dest, WHISPER_MARKERS):
            return dest
        raise RuntimeError(f"HuggingFace 下载不完整: {dest}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("HuggingFace 加载失败（%s），改从 ModelScope 下载", exc)
        _download_modelscope(ms_repo, dest)
        if not is_complete_model_dir(dest, WHISPER_MARKERS):
            raise RuntimeError(f"ModelScope 下载后仍缺少 Whisper 权重: {dest}") from exc
        return dest


def find_local_ocr_dir(name: str, models_dir: Path, override: Path | None = None) -> Path | None:
    if override is not None:
        path = resolve_user_path(str(override)) if not override.is_absolute() else override
        if is_complete_model_dir(path, OCR_MARKERS):
            return path
        raise FileNotFoundError(f"本地 OCR 目录无效或不完整: {path}")
    path = models_dir / name
    if is_complete_model_dir(path, OCR_MARKERS):
        return path
    return None


def _ensure_ocr_component(name: str, dest: Path) -> Path:
    if is_complete_model_dir(dest, OCR_MARKERS):
        return dest

    paddlex = _paddlex_dir(name)
    if is_complete_model_dir(paddlex, OCR_MARKERS):
        logger.info("从 PaddleX 缓存复制 OCR 模型到 %s", dest)
        copy_model_dir(paddlex, dest)
        return dest

    hf_repo = OCR_HF_REPOS[name]
    try:
        logger.info("从 HuggingFace 下载 OCR 模型到 %s", dest)
        _download_hf(hf_repo, dest)
        if is_complete_model_dir(dest, OCR_MARKERS):
            return dest
        raise RuntimeError(f"HuggingFace 下载不完整: {dest}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("HuggingFace OCR 下载失败（%s），改从 ModelScope 下载", exc)
        _download_modelscope(hf_repo, dest)
        if not is_complete_model_dir(dest, OCR_MARKERS):
            raise RuntimeError(f"无法准备 OCR 模型 {name}，请放到 {dest}") from exc
        return dest


def persist_paddlex_ocr(models_dir: Path, size: str = "small") -> None:
    """PaddleOCR 自行下载后，把 ~/.paddlex 缓存同步进项目 models/。"""
    for name in ocr_model_names(size):
        dest = models_dir / name
        if is_complete_model_dir(dest, OCR_MARKERS):
            continue
        src = _paddlex_dir(name)
        if is_complete_model_dir(src, OCR_MARKERS):
            logger.info("同步 PaddleX OCR 缓存到 %s", dest)
            copy_model_dir(src, dest)


def ensure_ocr_dirs(
    models_dir: Path | None = None,
    det_dir: Path | None = None,
    rec_dir: Path | None = None,
    size: str = "small",
) -> tuple[Path, Path]:
    models_dir = models_dir or MODELS_DIR
    det_name, rec_name = ocr_model_names(size)
    det = find_local_ocr_dir(det_name, models_dir, det_dir)
    rec = find_local_ocr_dir(rec_name, models_dir, rec_dir)
    if det is None:
        det = _ensure_ocr_component(det_name, models_dir / det_name)
    else:
        logger.info("使用本地 OCR 检测: %s", det)
    if rec is None:
        rec = _ensure_ocr_component(rec_name, models_dir / rec_name)
    else:
        logger.info("使用本地 OCR 识别: %s", rec)
    return det, rec


def ensure_sensevoice_dir(models_dir: Path | None = None) -> Path:
    models_dir = models_dir or MODELS_DIR
    dest = models_dir / SENSEVOICE_DIR_NAME
    if is_complete_model_dir(dest, SENSEVOICE_MARKERS):
        logger.info("使用本地 SenseVoice: %s", dest)
        return dest

    try:
        logger.info("本地无 SenseVoice，从 ModelScope 下载到 %s", dest)
        _download_modelscope(SENSEVOICE_MS_REPO, dest)
        if is_complete_model_dir(dest, SENSEVOICE_MARKERS):
            return dest
        raise RuntimeError(f"ModelScope 下载不完整: {dest}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ModelScope SenseVoice 下载失败（%s），改从 HuggingFace 下载", exc)
        _download_hf(SENSEVOICE_HF_REPO, dest)
        if not is_complete_model_dir(dest, SENSEVOICE_MARKERS):
            raise RuntimeError(f"无法准备 SenseVoice 权重，请放到 {dest}") from exc
        return dest
