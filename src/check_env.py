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

        _ok("faster_whisper", getattr(faster_whisper, "__version__", ""))
    except Exception as exc:  # noqa: BLE001
        _fail("faster_whisper", exc)
        failed += 1

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

    print("环境检查通过" if failed == 0 else f"有 {failed} 项未通过，先看 setup_env.sh 的安装日志")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
