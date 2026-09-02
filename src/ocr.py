from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _to_python(value.tolist())
        except Exception:  # noqa: BLE001
            return str(value)
    return value


def _as_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        if "res" in result and isinstance(result["res"], dict):
            return result["res"]
        return result
    if hasattr(result, "json"):
        data = result.json
        if isinstance(data, dict):
            if "res" in data and isinstance(data["res"], dict):
                return data["res"]
            return data
    if hasattr(result, "keys"):
        try:
            return dict(result)
        except Exception:  # noqa: BLE001
            pass
    return {}


def parse_ocr_result(raw: Any, min_score: float) -> list[dict[str, Any]]:
    """兼容 PaddleOCR 3.x predict 与 2.x ocr 两种返回。"""
    items: list[dict[str, Any]] = []
    if raw is None:
        return items

    # 3.x: list of result objects with rec_texts
    if isinstance(raw, list) and raw and not _looks_like_legacy_line(raw[0]):
        for entry in raw:
            data = _as_dict(entry)
            texts = data.get("rec_texts") or data.get("rec_text") or []
            scores = data.get("rec_scores") or data.get("rec_score") or []
            boxes = data.get("rec_polys") or data.get("rec_boxes") or data.get("dt_polys") or []
            if texts:
                for i, text in enumerate(texts):
                    score = None
                    if i < len(scores):
                        score = float(scores[i])
                    if score is not None and score < min_score:
                        continue
                    text_s = str(text).strip()
                    if not text_s:
                        continue
                    box = _to_python(boxes[i]) if i < len(boxes) else None
                    items.append({"text": text_s, "score": score, "box": box})
                continue
            items.extend(parse_legacy_ocr(entry, min_score))
        return items

    return parse_legacy_ocr(raw, min_score)


def _looks_like_legacy_line(entry: Any) -> bool:
    return (
        isinstance(entry, (list, tuple))
        and entry
        and isinstance(entry[0], (list, tuple))
        and len(entry[0]) >= 2
        and isinstance(entry[0][1], (list, tuple, str))
    )


def parse_legacy_ocr(raw: Any, min_score: float) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    lines = raw
    if isinstance(raw, list) and raw and isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list):
        # [[ [box, (text, score)], ... ]]  整页结果
        if not _looks_like_legacy_line(raw):
            for page in raw:
                items.extend(parse_legacy_ocr(page, min_score))
            return items
    if not isinstance(lines, list):
        return items
    for line in lines:
        if not line or not isinstance(line, (list, tuple)):
            continue
        box, payload = line[0], line[1] if len(line) > 1 else None
        text, score = "", None
        if isinstance(payload, (list, tuple)):
            text = str(payload[0]) if payload else ""
            if len(payload) > 1:
                score = float(payload[1])
        elif isinstance(payload, str):
            text = payload
        text = text.strip()
        if not text:
            continue
        if score is not None and score < min_score:
            continue
        items.append({"text": text, "score": score, "box": _to_python(box)})
    return items


class PaddleOCREngine:
    def __init__(self, lang: str = "ch", min_score: float = 0.5) -> None:
        from paddleocr import PaddleOCR

        logger.info("加载 PaddleOCR（lang=%s）", lang)
        kwargs = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        try:
            self.ocr = PaddleOCR(**kwargs)
        except TypeError:
            # 旧版 API
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        self.min_score = min_score
        self._use_predict = hasattr(self.ocr, "predict")

    def recognize_image(self, image: np.ndarray) -> list[dict[str, Any]]:
        raw = self._run(image)
        return parse_ocr_result(raw, self.min_score)

    def _run(self, image: np.ndarray) -> Any:
        if self._use_predict:
            try:
                return self.ocr.predict(image)
            except Exception:
                logger.debug("ndarray predict 失败，改为临时文件", exc_info=True)
                return self._run_via_tempfile(image, use_predict=True)
        return self._run_via_tempfile(image, use_predict=False)

    def _run_via_tempfile(self, image: np.ndarray, use_predict: bool) -> Any:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.close()
            cv2.imwrite(str(tmp_path), image)
            if use_predict:
                return self.ocr.predict(str(tmp_path))
            try:
                return self.ocr.ocr(str(tmp_path), cls=True)
            except TypeError:
                return self.ocr.ocr(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)
