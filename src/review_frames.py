"""把风险证据落到画面：抽问题帧，并按 OCR 框高亮问题文案。"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.media import extract_frame_at
from src.merge import normalize_key
from src.rules import RISK_LABELS

logger = logging.getLogger("review_frames")

_MAX_FRAMES = 6
_MIN_NEEDLE = 2
_SPACE_RE = re.compile(r"\s+")


def _compact(text: Any) -> str:
    return _SPACE_RE.sub("", str(text or "")).strip()


def box_to_pts(box: Any) -> np.ndarray | None:
    if not box or not isinstance(box, (list, tuple)):
        return None
    try:
        if box and isinstance(box[0], (list, tuple)):
            pts = [(float(p[0]), float(p[1])) for p in box if isinstance(p, (list, tuple)) and len(p) >= 2]
        else:
            nums = [float(x) for x in box]
            if len(nums) == 4:
                x1, y1, x2, y2 = nums
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            elif len(nums) >= 8:
                pts = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
            else:
                return None
    except (TypeError, ValueError):
        return None
    if len(pts) < 4:
        return None
    return np.array(pts[:8], dtype=np.float32)


def _needles_of(hit: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in ("matched", "evidence"):
        raw = str(hit.get(key) or "").strip()
        if not raw:
            continue
        pieces = [raw, _compact(raw), normalize_key(raw)]
        for piece in pieces:
            text = str(piece or "").strip()
            if len(text) < _MIN_NEEDLE or text in seen:
                continue
            seen.add(text)
            out.append(text)
    out.sort(key=len, reverse=True)
    return out


def _text_hits(ocr_text: str, needles: list[str]) -> bool:
    raw = str(ocr_text or "")
    compact = _compact(raw)
    key = normalize_key(raw)
    if len(key) < _MIN_NEEDLE:
        return False
    for needle in needles:
        ncomp = _compact(needle)
        nkey = normalize_key(needle)
        if ncomp and ncomp in compact:
            return True
        if nkey and nkey in key:
            return True
        if len(key) >= 4 and nkey and key in nkey:
            return True
    return False


def _nearest_ocr_frame(frames: list[dict[str, Any]], time_sec: float | None) -> dict[str, Any] | None:
    if not frames:
        return None
    if time_sec is None:
        return None
    best = None
    best_dist = 10**9
    for frame in frames:
        ts = frame.get("time")
        if not isinstance(ts, (int, float)):
            continue
        dist = abs(float(ts) - float(time_sec))
        if dist < best_dist:
            best = frame
            best_dist = dist
    if best is None or best_dist > 2.0:
        return None
    return best


def _collect_boxes(items: list[dict[str, Any]], needles: list[str]) -> list[np.ndarray]:
    boxes: list[np.ndarray] = []
    for item in items:
        if not _text_hits(str(item.get("text") or ""), needles):
            continue
        pts = box_to_pts(item.get("box"))
        if pts is not None:
            boxes.append(pts)
    return _merge_boxes(boxes)


def _merge_boxes(boxes: list[np.ndarray]) -> list[np.ndarray]:
    if len(boxes) <= 1:
        return boxes
    rects = [cv2.boundingRect(box.astype(np.int32)) for box in boxes]
    keep = [True] * len(boxes)
    for i, (x1, y1, w1, h1) in enumerate(rects):
        if not keep[i]:
            continue
        area1 = max(w1 * h1, 1)
        for j in range(i + 1, len(boxes)):
            if not keep[j]:
                continue
            x2, y2, w2, h2 = rects[j]
            inter_w = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            inter_h = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter = inter_w * inter_h
            area2 = max(w2 * h2, 1)
            if inter / (area1 + area2 - inter) < 0.3 and inter / max(area1, area2) < 0.7:
                continue
            if area2 > area1:
                keep[i] = False
                break
            keep[j] = False
    return [box for box, flag in zip(boxes, keep) if flag]


def select_problem_regions(
    ocr: dict[str, Any] | None,
    evidence: list[dict[str, Any]],
    *,
    max_frames: int = _MAX_FRAMES,
) -> list[dict[str, Any]]:
    """按证据挑出需要展示的画面时刻，并尽量带上 OCR 高亮框。"""
    frames = list((ocr or {}).get("frames") or [])
    merged = list((ocr or {}).get("merged") or [])
    hits = [h for h in evidence if isinstance(h, dict) and h.get("label") in RISK_LABELS]
    visual_hits = [h for h in hits if h.get("source") == "visual"]
    if not visual_hits:
        visual_hits = hits

    grouped: dict[str, dict[str, Any]] = {}

    def add_region(time_sec: float | None, hit: dict[str, Any], boxes: list[np.ndarray]) -> None:
        if time_sec is None:
            return
        key = f"{round(float(time_sec), 2):.2f}"
        row = grouped.get(key)
        if row is None:
            row = {
                "time": float(time_sec),
                "labels": [],
                "quotes": [],
                "boxes": [],
            }
            grouped[key] = row
        label = str(hit.get("label") or "")
        if label and label not in row["labels"]:
            row["labels"].append(label)
        quote = str(hit.get("matched") or hit.get("evidence") or "").strip()
        if quote and quote not in row["quotes"]:
            row["quotes"].append(quote)
        for box in boxes:
            row["boxes"].append(box.tolist())

    for hit in visual_hits:
        needles = _needles_of(hit)
        ts = hit.get("time")
        time_sec = float(ts) if isinstance(ts, (int, float)) else None
        frame = _nearest_ocr_frame(frames, time_sec)
        boxes: list[np.ndarray] = []
        if frame is not None:
            boxes = _collect_boxes(list(frame.get("texts") or []), needles)
            if time_sec is None and isinstance(frame.get("time"), (int, float)):
                time_sec = float(frame["time"])
        if not boxes:
            for cand in frames:
                found = _collect_boxes(list(cand.get("texts") or []), needles)
                if not found:
                    continue
                boxes = found
                if isinstance(cand.get("time"), (int, float)):
                    time_sec = float(cand["time"])
                break
        if time_sec is None:
            for line in merged:
                if not _text_hits(str(line.get("text") or ""), needles):
                    continue
                if isinstance(line.get("time"), (int, float)):
                    time_sec = float(line["time"])
                pts = box_to_pts(line.get("box"))
                if pts is not None:
                    boxes = [pts]
                break
        add_region(time_sec, hit, boxes)

    regions = sorted(grouped.values(), key=lambda r: r["time"])
    for row in regions:
        pts = [p for p in (box_to_pts(b) for b in row["boxes"]) if p is not None]
        row["boxes"] = [p.tolist() for p in _merge_boxes(pts)]
    return regions[: max(1, max_frames)]


def _scale_boxes(boxes: list[np.ndarray], image: np.ndarray) -> list[np.ndarray]:
    if not boxes:
        return []
    h, w = image.shape[:2]
    max_x = max(float(np.max(box[:, 0])) for box in boxes)
    max_y = max(float(np.max(box[:, 1])) for box in boxes)
    if max_x <= w * 1.05 and max_y <= h * 1.05:
        return [box.astype(np.int32) for box in boxes]
    sx = w / max(max_x, 1.0)
    sy = h / max(max_y, 1.0)
    return [(box * np.array([sx, sy], dtype=np.float32)).astype(np.int32) for box in boxes]


def annotate_frame(image: np.ndarray, boxes: list[Any]) -> np.ndarray:
    canvas = image.copy()
    pts_list = []
    for box in boxes:
        pts = box if isinstance(box, np.ndarray) else box_to_pts(box)
        if pts is not None:
            pts_list.append(pts)
    scaled = _scale_boxes(pts_list, canvas)
    if not scaled:
        return canvas
    overlay = canvas.copy()
    fill = (80, 91, 226)
    edge = (138, 196, 224)
    for pts in scaled:
        cv2.fillPoly(overlay, [pts], fill)
        cv2.polylines(canvas, [pts], True, edge, 3, cv2.LINE_AA)
        x, y, bw, bh = cv2.boundingRect(pts)
        pad = 6
        cv2.rectangle(
            canvas,
            (max(0, x - pad), max(0, y - pad)),
            (min(canvas.shape[1] - 1, x + bw + pad), min(canvas.shape[0] - 1, y + bh + pad)),
            edge,
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0, canvas)
    return canvas


def render_review_frames(
    video_path: Path,
    ocr: dict[str, Any] | None,
    evidence: list[dict[str, Any]],
    dest_dir: Path,
    *,
    max_width: int = 1280,
    max_frames: int = _MAX_FRAMES,
) -> list[dict[str, Any]]:
    """写出带高亮框的问题画面，返回可供前端展示的元数据。"""
    regions = select_problem_regions(ocr, evidence, max_frames=max_frames)
    if not regions:
        return []
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    for idx, region in enumerate(regions):
        image = extract_frame_at(video_path, region["time"], max_width=max_width)
        if image is None:
            logger.warning("抽帧失败 t=%.2f %s", region["time"], video_path.name)
            continue
        marked = annotate_frame(image, region.get("boxes") or [])
        name = f"t{int(round(region['time'] * 1000)):06d}.jpg"
        path = dest_dir / name
        ok = cv2.imwrite(str(path), marked, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            logger.warning("写入问题画面失败 %s", path)
            continue
        out.append(
            {
                "time": round(float(region["time"]), 3),
                "image_name": name,
                "labels": region["labels"],
                "quotes": region["quotes"],
                "highlighted": bool(region.get("boxes")),
                "index": idx,
            }
        )
    return out
