from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ok(name: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"[OK] {name}{suffix}")


def _fail(name: str, err: BaseException) -> None:
    print(f"[FAIL] {name}: {err}")


def main() -> int:
    print(f"Python {sys.version}")
    print(f"ROOT   {ROOT}")
    failed = 0

    try:
        from src.media import find_ffmpeg, probe_duration  # noqa: F401

        ffmpeg = find_ffmpeg()
        _ok("ffmpeg", ffmpeg)
    except Exception as exc:  # noqa: BLE001
        _fail("ffmpeg", exc)
        failed += 1

    for mod in ("cv2", "numpy", "pandas", "openpyxl", "tqdm"):
        try:
            loaded = importlib.import_module(mod)
            version = getattr(loaded, "__version__", "")
            _ok(mod, version)
        except Exception as exc:  # noqa: BLE001
            _fail(mod, exc)
            failed += 1

    try:
        import faster_whisper

        _ok("faster_whisper", getattr(faster_whisper, "__version__", "") + "（可选，非默认）")
    except Exception as exc:  # noqa: BLE001
        print(f"[INFO] faster_whisper 未安装（默认不用）: {exc}")

    try:
        import paddle

        _ok("paddle", paddle.__version__)
    except Exception as exc:  # noqa: BLE001
        _fail("paddle", exc)
        failed += 1

    try:
        from paddleocr import PaddleOCR  # noqa: F401

        _ok("paddleocr")
    except Exception as exc:  # noqa: BLE001
        _fail("paddleocr", exc)
        failed += 1

    try:
        import funasr

        _ok("funasr", getattr(funasr, "__version__", ""))
    except Exception as exc:  # noqa: BLE001
        _fail("funasr", exc)
        failed += 1

    try:
        from src.config import MODELS_DIR
        from src.local_models import (
            OCR_DET_NAME,
            OCR_REC_NAME,
            OCR_MARKERS,
            SENSEVOICE_DIR_NAME,
            SENSEVOICE_MARKERS,
            WHISPER_MARKERS,
            is_complete_model_dir,
        )

        whisper_dir = MODELS_DIR / "faster-whisper-small"
        det_dir = MODELS_DIR / OCR_DET_NAME
        rec_dir = MODELS_DIR / OCR_REC_NAME
        sense_dir = MODELS_DIR / SENSEVOICE_DIR_NAME
        if is_complete_model_dir(sense_dir, SENSEVOICE_MARKERS):
            _ok("local sensevoice", str(sense_dir))
        else:
            _fail("local sensevoice", FileNotFoundError(str(sense_dir)))
            failed += 1
        if is_complete_model_dir(det_dir, OCR_MARKERS) and is_complete_model_dir(rec_dir, OCR_MARKERS):
            _ok("local paddleocr", f"{det_dir.name}, {rec_dir.name}")
        else:
            _fail("local paddleocr", FileNotFoundError(f"{det_dir} / {rec_dir}"))
            failed += 1
        if is_complete_model_dir(whisper_dir, WHISPER_MARKERS):
            print(f"[INFO] 本地 Whisper 仍在（已不默认使用）: {whisper_dir}")
    except Exception as exc:  # noqa: BLE001
        _fail("local models", exc)
        failed += 1

    print("环境检查通过" if failed == 0 else f"有 {failed} 项未通过，先看 setup_env.sh 的安装日志")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
