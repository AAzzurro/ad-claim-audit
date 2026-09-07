"""E1：在已抽取的 ASR/OCR 上跑规则基线，写出结构化审核结果。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.asr import save_json
from src.config import ROOT
from src.merge import merge_ocr_frames
from src.rules import (
    LABEL_NORMAL,
    RISK_LABELS,
    hedge_nearby,
    iter_matches,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detect")

_SNIPPET = 24


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _snippet(text: str, start: int, end: int) -> str:
    lo = max(0, start - _SNIPPET)
    hi = min(len(text), end + _SNIPPET)
    piece = text[lo:hi].strip()
    if lo > 0:
        piece = "…" + piece
    if hi < len(text):
        piece = piece + "…"
    return piece


def _hits_in_text(
    text: str,
    source: str,
    *,
    time: float | None = None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for start, end, rule, surface in iter_matches(text):
        hedged = hedge_nearby(text, start, end)
        hits.append(
            {
                "label": rule.label,
                "evidence": _snippet(text, start, end),
                "matched": surface,
                "evidence_position": {
                    "source": source,
                    "time": time,
                    "span": [start, end],
                },
                "rule_basis": rule.rule_basis,
                "hedged": hedged,
            }
        )
    return hits


def detect_record(
    sample_id: str,
    asr: dict[str, Any] | None,
    ocr: dict[str, Any] | None,
) -> dict[str, Any]:
    asr = asr or {}
    ocr = ocr or {}
    hits: list[dict[str, Any]] = []

    segments = asr.get("segments") or []
    if segments:
        for seg in segments:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            start = seg.get("start")
            hits.extend(_hits_in_text(text, "audio", time=start if isinstance(start, (int, float)) else None))
    else:
        audio = str(asr.get("text") or "").strip()
        hits.extend(_hits_in_text(audio, "audio"))

    visual_lines = ocr.get("merged")
    if not visual_lines:
        visual_lines = merge_ocr_frames(ocr.get("frames") or [])
    for line in visual_lines:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        ts = line.get("time")
        hits.extend(
            _hits_in_text(text, "visual", time=ts if isinstance(ts, (int, float)) else None)
        )

    firm = [h for h in hits if not h["hedged"]]
    hedged_only = [h for h in hits if h["hedged"]]
    labels = []
    for hit in firm:
        if hit["label"] in RISK_LABELS and hit["label"] not in labels:
            labels.append(hit["label"])

    if labels:
        risk_labels = labels
        shown = firm
        explanation = "规则命中可见/可听原文；未根据本条视频增补词表。"
    elif hedged_only:
        risk_labels = [LABEL_NORMAL]
        shown = []
        explanation = "仅命中有条件描述附近的关键词，按任务书 9(3) 不标风险。"
    else:
        risk_labels = [LABEL_NORMAL]
        shown = []
        explanation = "现行法规/任务书词表未命中。未命中不等于人工认定无风险。"

    return {
        "sample_id": sample_id,
        "risk_labels": risk_labels,
        "evidence": [h["evidence"] for h in shown],
        "evidence_position": [h["evidence_position"] for h in shown],
        "rule_basis": list(dict.fromkeys(h["rule_basis"] for h in shown)),
        "hits": hits,
        "n_hits": len(firm),
        "n_hedged": len(hedged_only),
        "explanation": explanation,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1 关键词/规则基线（不跑 ASR/OCR）。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", default="", help="只处理这些 sample_id，逗号分隔")
    return parser.parse_args(argv)


def _wanted(only: str) -> set[str] | None:
    if not only.strip():
        return None
    keys: set[str] = set()
    for item in only.split(","):
        item = item.strip()
        if not item:
            continue
        keys.add(item)
        keys.add(Path(item).stem)
    return keys


def run(args: argparse.Namespace) -> int:
    asr_dir = args.data_dir / "asr"
    ocr_dir = args.data_dir / "ocr"
    out_dir = args.data_dir / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted({p.stem for p in asr_dir.glob("*.json")} | {p.stem for p in ocr_dir.glob("*.json")})
    wanted = _wanted(args.only)
    if wanted is not None:
        ids = [i for i in ids if i in wanted]
    if not ids:
        logger.warning("未找到可检测样本。先跑 python -m src.pipeline")
        return 0

    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        asr = _load_json(asr_dir / f"{sample_id}.json")
        ocr = _load_json(ocr_dir / f"{sample_id}.json")
        if ocr and ocr.get("frames"):
            ocr = {**ocr, "merged": merge_ocr_frames(ocr["frames"])}
        result = detect_record(sample_id, asr, ocr)
        save_json(out_dir / f"{sample_id}.json", result)
        rows.append(
            {
                "sample_id": sample_id,
                "risk_labels": result["risk_labels"],
                "n_hits": result["n_hits"],
            }
        )
        logger.info("%s %s hits=%s", sample_id, ",".join(result["risk_labels"]), result["n_hits"])

    index = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "e1",
        "n_samples": len(rows),
        "lexicon_note": "词表来自 Knowledge/ 任务书与法规，未用测试视频增补",
        "items": rows,
    }
    save_json(args.data_dir / "detect_index.json", index)
    logger.info("写成 %s", out_dir)
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
