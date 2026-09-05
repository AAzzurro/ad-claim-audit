"""E2：少样本结构化风险分类（小规模 LM，默认 Ollama Qwen2.5-7B）。"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.asr import save_json
from src.config import ROOT
from src.llm import DEFAULT_HOST, DEFAULT_MODEL, LLMError, chat_json, ensure_model
from src.merge import merge_ocr_frames
from src.prompts import ALLOWED_LABELS, FEW_SHOTS, SYSTEM_PROMPT
from src.rules import (
    LABEL_INDUCE,
    LABEL_NORMAL,
    LABEL_OFFSITE,
    LABEL_OTHER,
    LABEL_UNKNOWN,
    RISK_LABELS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classify")

_LABEL_ALIASES = {
    "夸大功效": "存在夸大功效",
    "功效夸大": "存在夸大功效",
    "虚假收益承诺": "存在虚假收益承诺",
    "收益承诺": "存在虚假收益承诺",
    "诱导消费": "存在诱导消费",
    "价格误导": "存在诱导消费",
    "促销误导": "存在诱导消费",
    "站外导流": "存在站外导流风险线索",
    "站外导流风险线索": "存在站外导流风险线索",
    "其他线索": "存在其他线索",
    "其他": "存在其他线索",
    "正常": LABEL_NORMAL,
    "无法判断": LABEL_UNKNOWN,
}
for _lab in ALLOWED_LABELS:
    _LABEL_ALIASES[_lab] = _lab


_INDUCE_EV = re.compile(
    r"最后|仅此|错过不|最低价|历史最低|全年最低|全网最低|倒计时|仅剩|只剩|"
    r"活动就|卖完|断货|过期不候|名额|不买就"
)
_OFFSITE_EV = re.compile(r"扫码|加群|加微信|加薇|加v\b|加V\b|私信|淘口令|站外联系")
_OTHER_EV = re.compile(r"好运|转运|开运|迷信|风水")


def _evidence_supports(label: str, text: str) -> bool:
    """核验证据短句是否撑得起该标签，避免把领券/拍链接误当成风险。"""
    if not text:
        return False
    if label == LABEL_INDUCE:
        return bool(_INDUCE_EV.search(text))
    if label == LABEL_OFFSITE:
        return bool(_OFFSITE_EV.search(text))
    if label == LABEL_OTHER:
        return bool(_OTHER_EV.search(text))
    return True


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _wanted(only: str) -> set[str] | None:
    if not only.strip():
        return None
    keys: set[str] = set()
    for item in only.split(","):
        item = item.strip()
        if item:
            keys.add(item)
            keys.add(Path(item).stem)
    return keys


def build_user_prompt(asr: dict[str, Any] | None, ocr: dict[str, Any] | None) -> tuple[str, str, list[dict[str, Any]]]:
    asr = asr or {}
    ocr = ocr or {}
    audio = str(asr.get("text") or "").strip()
    lines = ocr.get("merged")
    if not lines:
        lines = merge_ocr_frames(ocr.get("frames") or [])
    visual_bits: list[str] = []
    for line in lines:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        ts = line.get("time")
        if isinstance(ts, (int, float)):
            visual_bits.append(f"[{ts:.1f}s] {text}")
        else:
            visual_bits.append(text)
    visual_block = "\n".join(visual_bits)
    if len(visual_block) > 1800:
        visual_block = visual_block[:1800] + "\n…"
    if len(audio) > 1200:
        audio = audio[:1200] + "…"
    prompt = f"【口播 ASR】\n{audio}\n【画面 OCR】\n{visual_block}"
    return prompt, audio, lines


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("输出不是 JSON 对象")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON 根节点不是对象")
    return obj


def _canon_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in _LABEL_ALIASES:
        return _LABEL_ALIASES[text]
    for key, mapped in _LABEL_ALIASES.items():
        if key in text or text in key:
            return mapped
    return None


def _normalize_labels(raw: Any) -> list[str]:
    if raw is None:
        values: list[Any] = []
    elif isinstance(raw, str):
        values = re.split(r"[；;,/|]", raw)
    else:
        values = list(raw)
    labels: list[str] = []
    for item in values:
        mapped = _canon_label(item)
        if mapped and mapped not in labels:
            labels.append(mapped)
    risks = [x for x in labels if x in RISK_LABELS]
    if risks:
        return [x for x in RISK_LABELS if x in risks]
    if LABEL_UNKNOWN in labels:
        return [LABEL_UNKNOWN]
    if LABEL_NORMAL in labels:
        return [LABEL_NORMAL]
    return [LABEL_UNKNOWN]


def _align_labels_with_evidence(labels: list[str], hits: list[dict[str, Any]]) -> list[str]:
    """风险标签必须能对上一条已落地的原文证据。"""
    ev_labels = [h["label"] for h in hits if h.get("label") in RISK_LABELS and h.get("grounded")]
    if any(lab in RISK_LABELS for lab in labels) or ev_labels:
        aligned = [lab for lab in RISK_LABELS if lab in labels and lab in ev_labels]
        if not aligned:
            aligned = [lab for lab in RISK_LABELS if lab in ev_labels]
        return aligned or [LABEL_NORMAL]
    if LABEL_UNKNOWN in labels:
        return [LABEL_UNKNOWN]
    return [LABEL_NORMAL]


def _find_span(haystack: str, needle: str) -> tuple[int, int] | None:
    if not haystack or not needle:
        return None
    idx = haystack.find(needle)
    if idx >= 0:
        return idx, idx + len(needle)
    if len(needle) >= 6:
        for n in range(len(needle), 3, -1):
            piece = needle[:n]
            idx = haystack.find(piece)
            if idx >= 0:
                return idx, idx + len(piece)
    return None


def ground_evidence(
    items: Any,
    audio: str,
    visual_lines: list[dict[str, Any]],
    labels: list[str],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    visual_text = " ".join(str(x.get("text") or "") for x in visual_lines)
    rows: list[dict[str, Any]] = []
    if not isinstance(items, list):
        items = []
    for item in items:
        if not isinstance(item, dict):
            text = str(item).strip()
            label = labels[0] if labels else LABEL_UNKNOWN
            source_hint = ""
        else:
            text = str(item.get("text") or item.get("evidence") or "").strip()
            label = _canon_label(item.get("label")) or (labels[0] if labels else LABEL_UNKNOWN)
            source_hint = str(item.get("source") or "").strip().lower()
        if not text:
            continue
        pos: dict[str, Any] = {"source": "unknown", "time": None, "span": None}
        span = None
        if source_hint != "visual":
            span = _find_span(audio, text)
            if span:
                pos = {"source": "audio", "time": 0.0 if audio else None, "span": list(span)}
        if pos["source"] == "unknown":
            for line in visual_lines:
                line_text = str(line.get("text") or "")
                span = _find_span(line_text, text)
                if span:
                    ts = line.get("time")
                    pos = {
                        "source": "visual",
                        "time": ts if isinstance(ts, (int, float)) else None,
                        "span": list(span),
                    }
                    break
            if pos["source"] == "unknown":
                span = _find_span(visual_text, text)
                if span:
                    pos = {"source": "visual", "time": None, "span": list(span)}
        if pos["source"] == "unknown" and source_hint == "visual":
            span = _find_span(audio, text)
            if span:
                pos = {"source": "audio", "time": 0.0, "span": list(span)}
        rows.append(
            {
                "label": label if label in RISK_LABELS else (label or LABEL_UNKNOWN),
                "evidence": text,
                "matched": text,
                "evidence_position": pos,
                "rule_basis": "",
                "hedged": False,
                "grounded": pos["source"] != "unknown",
            }
        )
    kept = []
    for row in rows:
        if row["label"] in RISK_LABELS and not _evidence_supports(row["label"], row["evidence"]):
            continue
        kept.append(row)
    evidence = [r["evidence"] for r in kept]
    positions = [r["evidence_position"] for r in kept]
    return evidence, positions, kept


def classify_record(
    sample_id: str,
    asr: dict[str, Any] | None,
    ocr: dict[str, Any] | None,
    *,
    model: str,
    host: str,
) -> dict[str, Any]:
    user_prompt, audio, visual_lines = build_user_prompt(asr, ocr)
    if not audio and not any(str(x.get("text") or "").strip() for x in visual_lines):
        return {
            "sample_id": sample_id,
            "risk_labels": [LABEL_UNKNOWN],
            "evidence": [],
            "evidence_position": [],
            "rule_basis": ["任务书 9(5) 证据不足时应输出无法判断"],
            "hits": [],
            "n_hits": 0,
            "n_hedged": 0,
            "explanation": "口播与画面均无可用原文。",
            "model": model,
            "raw": "",
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in FEW_SHOTS:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_prompt})

    raw = chat_json(messages, model=model, host=host)
    parsed = _extract_json(raw)
    labels = _normalize_labels(parsed.get("risk_labels") or parsed.get("labels"))
    proposed = labels
    evidence, positions, hits = ground_evidence(parsed.get("evidence"), audio, visual_lines, labels)
    labels = _align_labels_with_evidence(labels, hits)
    evidence = [h["evidence"] for h in hits]
    positions = [h["evidence_position"] for h in hits]
    basis = parsed.get("rule_basis") or []
    if isinstance(basis, str):
        basis = [basis]
    basis = [str(x).strip() for x in basis if str(x).strip()]
    explanation = str(parsed.get("explanation") or "").strip()
    if proposed != labels:
        if labels == [LABEL_NORMAL]:
            explanation = "模型给出的风险缺少合格原文证据，按正常处理。"
        elif explanation:
            explanation = explanation.rstrip("。") + "。已去掉缺少合格证据的标签。"
    if labels == [LABEL_UNKNOWN] and not explanation:
        explanation = "模型认为证据不足。"
    if labels == [LABEL_NORMAL] and not explanation:
        explanation = "未见可核的虚假宣传原文。"
    for hit in hits:
        if basis and not hit["rule_basis"]:
            hit["rule_basis"] = basis[0]
    return {
        "sample_id": sample_id,
        "risk_labels": labels,
        "evidence": evidence,
        "evidence_position": positions,
        "rule_basis": basis,
        "hits": hits,
        "n_hits": len(hits),
        "n_hedged": 0,
        "explanation": explanation,
        "model": model,
        "raw": raw,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2 少样本结构化分类（Ollama 小模型，不跑 ASR/OCR）。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", default="", help="只处理这些 sample_id，逗号分隔")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 模型名，默认 qwen2.5:7b")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--force", action="store_true", help="覆盖已有分类结果")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    asr_dir = args.data_dir / "asr"
    ocr_dir = args.data_dir / "ocr"
    out_dir = args.data_dir / "classify"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_model(args.model, args.host)
    except LLMError as exc:
        logger.error("%s", exc)
        return 1

    ids = sorted({p.stem for p in asr_dir.glob("*.json")} | {p.stem for p in ocr_dir.glob("*.json")})
    wanted = _wanted(args.only)
    if wanted is not None:
        ids = [i for i in ids if i in wanted]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        logger.warning("未找到可分类样本。先跑 python -m src.pipeline")
        return 0

    rows: list[dict[str, Any]] = []
    failures = 0
    for sample_id in ids:
        dest = out_dir / f"{sample_id}.json"
        if dest.exists() and not args.force:
            prev = _load_json(dest) or {}
            rows.append(
                {
                    "sample_id": sample_id,
                    "risk_labels": prev.get("risk_labels") or [],
                    "n_hits": prev.get("n_hits", 0),
                    "skipped": True,
                }
            )
            logger.info("%s skip", sample_id)
            continue
        asr = _load_json(asr_dir / f"{sample_id}.json")
        ocr = _load_json(ocr_dir / f"{sample_id}.json")
        if ocr and ocr.get("frames"):
            ocr = {**ocr, "merged": merge_ocr_frames(ocr["frames"])}
        try:
            result = classify_record(sample_id, asr, ocr, model=args.model, host=args.host)
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            failures += 1
            logger.exception("分类失败 %s", sample_id)
            result = {
                "sample_id": sample_id,
                "risk_labels": [LABEL_UNKNOWN],
                "evidence": [],
                "evidence_position": [],
                "rule_basis": [],
                "hits": [],
                "n_hits": 0,
                "n_hedged": 0,
                "explanation": f"模型调用或解析失败：{exc}",
                "model": args.model,
                "raw": "",
            }
        save_json(dest, result)
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
        "experiment": "e2",
        "model": args.model,
        "n_samples": len(rows),
        "n_failed": failures,
        "prompt_note": "少样本示例来自任务书与法规，未用测试视频原文",
        "items": rows,
    }
    save_json(args.data_dir / "classify_index.json", index)
    logger.info("写成 %s", out_dir)
    return 1 if failures else 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
