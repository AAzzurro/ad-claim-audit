"""E3：把已有 E1/E2 预测按标签并集融合。不重跑模型，不加测试集词。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analyze import _fuse
from src.asr import save_json
from src.config import ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hybrid")


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fuse_record(sample_id: str, e1: dict[str, Any], e2: dict[str, Any]) -> dict[str, Any]:
    fused = _fuse(e1, e2)
    return {
        "sample_id": sample_id,
        "experiment": "e3",
        "risk_labels": fused["risk_labels"],
        "evidence": [h.get("evidence") for h in fused.get("hits") or [] if h.get("evidence")],
        "evidence_position": [
            {
                "source": h.get("source"),
                "time": h.get("time"),
                "span": h.get("span"),
                "engine": h.get("engine"),
            }
            for h in fused.get("hits") or []
        ],
        "rule_basis": fused.get("rule_basis") or [],
        "explanation": fused.get("explanation") or "",
        "hits": fused.get("hits") or [],
        "n_hits": fused.get("n_hits") or 0,
        "n_hedged": fused.get("n_hedged") or 0,
        "engines": {
            "rules": {"risk_labels": e1.get("risk_labels") or []},
            "model": {"risk_labels": e2.get("risk_labels") or []},
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E3：融合 data/detect 与 data/classify。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--detect-dir", type=Path, default=None)
    parser.add_argument("--classify-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    detect_dir = args.detect_dir or args.data_dir / "detect"
    classify_dir = args.classify_dir or args.data_dir / "classify"
    out_dir = args.out_dir or args.data_dir / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = sorted(
        {p.stem for p in detect_dir.glob("*.json") if not p.name.startswith(".")}
        & {p.stem for p in classify_dir.glob("*.json") if not p.name.startswith(".")}
    )
    if not ids:
        logger.error("没有可融合的 E1/E2 结果")
        return 1

    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        e1 = _load(detect_dir / f"{sample_id}.json")
        e2 = _load(classify_dir / f"{sample_id}.json")
        if not e1 or not e2:
            continue
        result = fuse_record(sample_id, e1, e2)
        save_json(out_dir / f"{sample_id}.json", result)
        rows.append({"sample_id": sample_id, "risk_labels": result["risk_labels"], "n_hits": result["n_hits"]})

    save_json(
        args.data_dir / "hybrid_index.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "e3",
            "n_samples": len(rows),
            "note": "E1∪E2 标签并集；不重跑 LLM，不按测试集改词表",
            "items": rows,
        },
    )
    logger.info("融合 %s 条 → %s", len(rows), out_dir)
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
