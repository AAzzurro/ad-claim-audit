"""E3：把已有 E1/E2 预测按标签并集融合。不重跑模型，不加测试集词。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analyze import _fuse, _fuse_many
from src.asr import save_json
from src.config import ROOT
from src.rules import finalize_labels

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


def fuse_record(
    sample_id: str,
    e1: dict[str, Any],
    e2: dict[str, Any],
    *,
    experiment: str = "e3",
) -> dict[str, Any]:
    fused = _fuse(e1, e2)
    return _pack_fused(
        sample_id,
        fused,
        experiment=experiment,
        engines={
            "rules": {"risk_labels": finalize_labels(e1.get("risk_labels") or [])},
            "model": {"risk_labels": finalize_labels(e2.get("risk_labels") or [])},
        },
    )


def fuse_named(sample_id: str, named: list[tuple[str, dict[str, Any]]], *, experiment: str) -> dict[str, Any]:
    fused = _fuse_many(named)
    return _pack_fused(
        sample_id,
        fused,
        experiment=experiment,
        engines={name: {"risk_labels": finalize_labels(rec.get("risk_labels") or [])} for name, rec in named},
    )


def _pack_fused(
    sample_id: str,
    fused: dict[str, Any],
    *,
    experiment: str,
    engines: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "experiment": experiment,
        "risk_labels": finalize_labels(fused["risk_labels"]),
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
        "engines": engines,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="融合规则与一路或多路模型预测。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--detect-dir", type=Path, default=None)
    parser.add_argument(
        "--classify-dir",
        action="append",
        default=None,
        help="可重复。默认一路 data/classify；多路时按出现顺序命名 model / model_2 …",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="e3")
    parser.add_argument("--note", default="E1∪E2 标签并集；不重跑 LLM，不按测试集改词表")
    parser.add_argument("--index-name", default="hybrid_index.json")
    parser.add_argument(
        "--engine-name",
        action="append",
        default=None,
        help="与每个 --classify-dir 对应的引擎名，例如 model_audio",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    detect_dir = args.detect_dir or args.data_dir / "detect"
    classify_dirs = [Path(p) for p in (args.classify_dir or [args.data_dir / "classify"])]
    engine_names = list(args.engine_name or [])
    while len(engine_names) < len(classify_dirs):
        engine_names.append("model" if len(engine_names) == 0 else f"model_{len(engine_names) + 1}")
    out_dir = args.out_dir or args.data_dir / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)

    id_sets = [{p.stem for p in detect_dir.glob("*.json") if not p.name.startswith(".")}]
    for folder in classify_dirs:
        id_sets.append({p.stem for p in folder.glob("*.json") if not p.name.startswith(".")})
    ids = sorted(set.intersection(*id_sets)) if id_sets else []
    if not ids:
        logger.error("没有可融合的预测")
        return 1

    rows: list[dict[str, Any]] = []
    for sample_id in ids:
        e1 = _load(detect_dir / f"{sample_id}.json")
        if not e1:
            continue
        named: list[tuple[str, dict[str, Any]]] = [("rules", e1)]
        missing = False
        for folder, name in zip(classify_dirs, engine_names):
            rec = _load(folder / f"{sample_id}.json")
            if not rec:
                missing = True
                break
            named.append((name, rec))
        if missing:
            continue
        result = fuse_named(sample_id, named, experiment=args.experiment)
        save_json(out_dir / f"{sample_id}.json", result)
        rows.append({"sample_id": sample_id, "risk_labels": result["risk_labels"], "n_hits": result["n_hits"]})

    save_json(
        args.data_dir / args.index_name,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": args.experiment,
            "n_samples": len(rows),
            "note": args.note,
            "sources": [str(detect_dir)] + [str(p) for p in classify_dirs],
            "items": rows,
        },
    )
    logger.info("融合 %s 条 → %s", len(rows), out_dir)
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
