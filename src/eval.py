"""对照人工标注表计算多标签 Precision / Recall / F1。

默认评 E1（data/detect）。不根据评测结果改 src/rules.py 词表。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.asr import save_json
from src.config import ROOT
from src.rules import (
    LABEL_NORMAL,
    RISK_LABELS,
    finalize_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")

ALL_LABELS = (*RISK_LABELS, LABEL_NORMAL)


def _parse_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(x).strip() for x in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        items = [p.strip() for p in text.replace(",", "；").split("；")]
    return [x for x in items if x]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_gold(path: Path) -> dict[str, list[str]]:
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        import pandas as pd

        sheet = "人工标注表" if path.suffix.lower() in {".xlsx", ".xls"} else None
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_csv(path)
        rows = df.to_dict(orient="records")
    gold: dict[str, list[str]] = {}
    for row in rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        gold[sid] = finalize_labels(_parse_labels(row.get("风险标签")))
    return gold


def load_preds(detect_dir: Path, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
    preds: dict[str, dict[str, Any]] = {}
    for path in sorted(detect_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        sid = str(rec.get("sample_id") or path.stem)
        if only is not None and sid not in only:
            continue
        rec["risk_labels"] = finalize_labels(_parse_labels(rec.get("risk_labels")))
        preds[sid] = rec
    return preds


def evaluate(
    gold: dict[str, list[str]],
    preds: dict[str, dict[str, Any]],
    experiment: str,
) -> dict[str, Any]:
    ids = sorted(gold, key=lambda x: (len(x), x))
    missing = [sid for sid in ids if sid not in preds]
    extra = sorted(set(preds) - set(gold))
    scored = [sid for sid in ids if sid in preds]

    counts: dict[str, dict[str, int]] = {
        lab: {"tp": 0, "fp": 0, "fn": 0} for lab in ALL_LABELS
    }
    exact = 0
    disagreements: list[dict[str, Any]] = []

    for sid in scored:
        g = set(gold[sid])
        p = set(preds[sid]["risk_labels"])
        if g == p:
            exact += 1
        else:
            pred_rec = preds[sid]
            disagreements.append(
                {
                    "sample_id": sid,
                    "gold": "；".join(gold[sid]),
                    "pred": "；".join(pred_rec["risk_labels"]),
                    "n_hits": pred_rec.get("n_hits", 0),
                    "evidence": " || ".join(pred_rec.get("evidence") or [])[:240],
                    "missed": "；".join(x for x in gold[sid] if x not in p),
                    "extra": "；".join(x for x in pred_rec["risk_labels"] if x not in g),
                }
            )
        for lab in ALL_LABELS:
            in_g = lab in g
            in_p = lab in p
            if in_g and in_p:
                counts[lab]["tp"] += 1
            elif in_p and not in_g:
                counts[lab]["fp"] += 1
            elif in_g and not in_p:
                counts[lab]["fn"] += 1

    per_label = {lab: _prf(**counts[lab]) for lab in ALL_LABELS}
    risk_rows = [per_label[lab] for lab in RISK_LABELS]
    macro = {
        "precision": round(sum(r["precision"] for r in risk_rows) / len(risk_rows), 4),
        "recall": round(sum(r["recall"] for r in risk_rows) / len(risk_rows), 4),
        "f1": round(sum(r["f1"] for r in risk_rows) / len(risk_rows), 4),
    }
    micro_tp = sum(counts[lab]["tp"] for lab in RISK_LABELS)
    micro_fp = sum(counts[lab]["fp"] for lab in RISK_LABELS)
    micro_fn = sum(counts[lab]["fn"] for lab in RISK_LABELS)
    micro = _prf(micro_tp, micro_fp, micro_fn)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "n_gold": len(gold),
        "n_pred": len(preds),
        "n_scored": len(scored),
        "n_missing_pred": len(missing),
        "n_extra_pred": len(extra),
        "missing_pred_ids": missing,
        "exact_match": round(exact / len(scored), 4) if scored else 0.0,
        "n_exact": exact,
        "n_disagree": len(disagreements),
        "per_label": per_label,
        "macro_risk": macro,
        "micro_risk": {k: micro[k] for k in ("precision", "recall", "f1", "support")},
        "note": (
            "词表冻结：本表只作事后评测，不回写规则。"
            if experiment == "e1"
            else "E1∪E2 融合事后评测；未按测试集改词表或提示词。"
            if experiment == "e3"
            else "少样本分类事后评测；提示词示例来自任务书与法规，未用测试视频。"
        ),
        "disagreements": disagreements,
    }


def _print_table(report: dict[str, Any]) -> None:
    print(f"scored={report['n_scored']} exact_match={report['exact_match']}")
    print(f"macro_risk  P={report['macro_risk']['precision']} R={report['macro_risk']['recall']} F1={report['macro_risk']['f1']}")
    print(f"micro_risk  P={report['micro_risk']['precision']} R={report['micro_risk']['recall']} F1={report['micro_risk']['f1']}")
    print(f"{'label':<20} {'P':>7} {'R':>7} {'F1':>7} {'sup':>5} {'tp':>4} {'fp':>4} {'fn':>4}")
    for lab in ALL_LABELS:
        row = report["per_label"][lab]
        print(
            f"{lab:<20} {row['precision']:7.4f} {row['recall']:7.4f} {row['f1']:7.4f} "
            f"{row['support']:5d} {row['tp']:4d} {row['fp']:4d} {row['fn']:4d}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对照人工标注计算 P/R/F1。")
    parser.add_argument("--gold", type=Path, default=ROOT / "annotations" / "manual_annotation.json")
    parser.add_argument("--detect-dir", type=Path, default=ROOT / "data" / "detect")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "eval")
    parser.add_argument("--experiment", default="e1")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    gold = load_gold(args.gold)
    preds = load_preds(args.detect_dir)
    if not gold:
        logger.error("标注表为空：%s", args.gold)
        return 1
    if not preds:
        logger.error("没有预测结果：%s  （先跑 python -m src.detect）", args.detect_dir)
        return 1

    report = evaluate(gold, preds, args.experiment)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in report.items() if k != "disagreements"}
    save_json(args.out_dir / f"{args.experiment}_metrics.json", slim)

    csv_path = args.out_dir / f"{args.experiment}_disagreements.csv"
    fields = ["sample_id", "gold", "pred", "n_hits", "missed", "extra", "evidence"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["disagreements"])

    _print_table(report)
    e1_path = args.out_dir / "e1_metrics.json"
    if args.experiment != "e1" and e1_path.exists():
        e1 = json.loads(e1_path.read_text(encoding="utf-8"))
        print("--- vs E1 ---")
        for name, key in (("micro_f1", "micro_risk"), ("macro_f1", "macro_risk")):
            old = e1.get(key, {}).get("f1")
            new = report.get(key, {}).get("f1")
            print(f"{name}: e1={old} {args.experiment}={new}")
        print(f"exact_match: e1={e1.get('exact_match')} {args.experiment}={report['exact_match']}")
    logger.info("写成 %s 和 %s", args.out_dir / f"{args.experiment}_metrics.json", csv_path)
    if report["n_missing_pred"]:
        logger.warning("缺预测 %s 条：%s", report["n_missing_pred"], report["missing_pred_ids"][:12])
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
