"""E4b 第一步：从 ASR/OCR 抽出原子宣称，再对清单做少样本分类。

不改词表、不加测试集示例。模型仍是 Ollama Qwen2.5-7B。
宣称级判定不再用整段分类的词表二次否决，否则无法补词表外的收益承诺。
"""

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
from src.classify import (
    _extract_json,
    _failed_record,
    _load_json,
    _wanted,
    assemble_record,
    build_user_prompt,
)
from src.config import ROOT
from src.llm import DEFAULT_HOST, DEFAULT_MODEL, LLMError, chat_json, ensure_model
from src.merge import merge_ocr_frames
from src.prompts import CLAIM_CLASSIFY_PROMPT, CLAIM_CLASSIFY_SHOTS, CLAIM_EXTRACT_PROMPT, CLAIM_EXTRACT_SHOTS
from src.rules import LABEL_NORMAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("claims")

_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_VENUE_ONLY = re.compile(r"博览会|号馆|作品列表|巡展|开幕|太原站|会展")
_DATE_ONLY = re.compile(r"^\d{1,4}\s*月|\d+日-\d+日|20\d{2}")
# 任务书/法规里的风险线索，用来给候选行排序，不是测试集原句。
_CUE = re.compile(
    r"天花板|根治|治愈|速效|治疗|首创|首个|首款|唯一|第一|最佳|最高级|"
    r"稳赚|保本|无风险|日入|包过|保过|躺赚|"
    r"最后几单|仅此一次|错过不再|最低价|全年最低|全网最低"
)
MAX_CLAIMS = 16
MAX_CANDIDATES = 28


def _has_cjk(text: str, min_chars: int = 2) -> bool:
    return len(_CJK.findall(text)) >= min_chars


def build_candidates(asr: dict[str, Any] | None, ocr: dict[str, Any] | None) -> list[dict[str, Any]]:
    """口播分句 + 画面叠字，供抽取模型点选。不是词表。"""
    asr = asr or {}
    ocr = ocr or {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    audio = str(asr.get("text") or "").strip()
    for piece in _SENT_SPLIT.split(audio):
        text = piece.strip(" \t，,")
        if not _has_cjk(text) or len(text) < 4:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "source": "audio"})

    lines = ocr.get("merged")
    if not lines:
        lines = merge_ocr_frames(ocr.get("frames") or [])
    for line in lines:
        text = str(line.get("text") or "").strip()
        if not _has_cjk(text) or len(text) < 3 or len(text) > 60:
            continue
        if _VENUE_ONLY.search(text) and not _CUE.search(text):
            continue
        if _DATE_ONLY.search(text) and not _CUE.search(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "source": "visual", "time": line.get("time")})

    return rows[:MAX_CANDIDATES]


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return ""
    lines = []
    for i, row in enumerate(candidates, 1):
        lines.append(f"{i}. [{row.get('source') or 'unknown'}] {row['text']}")
    return "\n".join(lines)


def _ground_claim(text: str, source_hint: str, audio: str, visual_lines: list[dict[str, Any]]) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    source_hint = (source_hint or "").strip().lower()
    if source_hint != "visual":
        idx = audio.find(text)
        if idx >= 0:
            return {"text": text, "source": "audio", "time": 0.0 if audio else None, "span": [idx, idx + len(text)]}
    for line in visual_lines:
        line_text = str(line.get("text") or "")
        idx = line_text.find(text)
        if idx >= 0:
            ts = line.get("time")
            return {
                "text": text,
                "source": "visual",
                "time": ts if isinstance(ts, (int, float)) else None,
                "span": [idx, idx + len(text)],
            }
    visual_text = " ".join(str(x.get("text") or "") for x in visual_lines)
    idx = visual_text.find(text)
    if idx >= 0:
        return {"text": text, "source": "visual", "time": None, "span": [idx, idx + len(text)]}
    if source_hint == "visual":
        idx = audio.find(text)
        if idx >= 0:
            return {"text": text, "source": "audio", "time": 0.0, "span": [idx, idx + len(text)]}
    return None


def normalize_claims(
    raw: Any,
    audio: str,
    visual_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items = raw if isinstance(raw, list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("claim") or "").strip()
            source = str(item.get("source") or "")
        else:
            text = str(item).strip()
            source = ""
        grounded = _ground_claim(text, source, audio, visual_lines)
        if not grounded:
            continue
        key = grounded["text"]
        if key in seen:
            continue
        seen.add(key)
        out.append(grounded)
        if len(out) >= MAX_CLAIMS:
            break
    return out


def _rank_candidate(row: dict[str, Any]) -> tuple[int, int, int]:
    text = str(row.get("text") or "")
    cue = 0 if _CUE.search(text) else 1
    visual = 0 if row.get("source") == "visual" else 1
    n = len(text)
    length_pen = 0 if 6 <= n <= 28 else (1 if n <= 40 else 2)
    return (cue, visual, length_pen)


def merge_claim_lists(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for group in groups:
        for claim in group:
            key = str(claim.get("text") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(claim)
            if len(out) >= MAX_CLAIMS:
                return out
    return out


def heuristic_claims(candidates: list[dict[str, Any]], audio: str, visual_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """结构化抽取：口播分句 + 画面叠字，按法规线索排序后落地。"""
    ranked = sorted(candidates, key=_rank_candidate)
    return normalize_claims(
        [{"text": c["text"], "source": c.get("source") or ""} for c in ranked],
        audio,
        visual_lines,
    )[:MAX_CLAIMS]


def extract_claims(
    asr: dict[str, Any] | None,
    ocr: dict[str, Any] | None,
    *,
    model: str,
    host: str,
    retries: int = 1,
) -> dict[str, Any]:
    user_prompt, audio, visual_lines = build_user_prompt(asr, ocr)
    candidates = build_candidates(asr, ocr)
    cand_block = _format_candidates(candidates)
    user_prompt = f"{user_prompt}\n【候选行】\n{cand_block}"

    if not audio and not any(str(x.get("text") or "").strip() for x in visual_lines):
        return {"claims": [], "candidates": [], "raw": "", "fallback": False, "model": model}

    messages = [{"role": "system", "content": CLAIM_EXTRACT_PROMPT}]
    for user, assistant in CLAIM_EXTRACT_SHOTS:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_prompt})

    attempts = max(1, retries + 1)
    last_raw = ""
    last_exc: BaseException | None = None
    llm_claims: list[dict[str, Any]] = []
    for attempt in range(attempts):
        try:
            raw = chat_json(
                messages,
                model=model,
                host=host,
                temperature=0.0 if attempt == 0 else 0.2,
                num_predict=512,
            )
            last_raw = raw
            parsed = _extract_json(raw)
            llm_claims = normalize_claims(parsed.get("claims"), audio, visual_lines)
            break
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                logger.warning("宣称抽取不合法，重试 %s/%s：%s", attempt + 1, retries, exc)
            else:
                logger.warning("宣称抽取失败：%s", exc)

    structured = heuristic_claims(candidates, audio, visual_lines)
    parsed_claims = merge_claim_lists(structured, llm_claims)
    fallback = bool(structured) and (not llm_claims or len(parsed_claims) > len(llm_claims))
    if fallback:
        logger.info("结构化补全宣称 %s→%s", len(llm_claims), len(parsed_claims))

    return {
        "claims": parsed_claims,
        "candidates": [{"text": c["text"], "source": c.get("source")} for c in candidates],
        "n_llm": len(llm_claims),
        "raw": last_raw,
        "fallback": fallback,
        "error": str(last_exc) if last_exc and not parsed_claims else "",
        "model": model,
    }


def _claims_user_prompt(claims: list[dict[str, Any]]) -> str:
    if not claims:
        return "【宣称清单】\n（空）"
    lines = ["【宣称清单】"]
    for i, claim in enumerate(claims, 1):
        lines.append(f"{i}. [{claim.get('source') or 'unknown'}] {claim['text']}")
    return "\n".join(lines)


def classify_claims(
    sample_id: str,
    claims: list[dict[str, Any]],
    asr: dict[str, Any] | None,
    ocr: dict[str, Any] | None,
    *,
    model: str,
    host: str,
    retries: int = 1,
) -> dict[str, Any]:
    _, audio, visual_lines = build_user_prompt(asr, ocr)
    if not claims:
        return {
            "sample_id": sample_id,
            "risk_labels": [LABEL_NORMAL],
            "evidence": [],
            "evidence_position": [],
            "rule_basis": ["未见可核宣称，按正常处理"],
            "hits": [],
            "n_hits": 0,
            "n_hedged": 0,
            "explanation": "未抽出可核宣称。",
            "model": model,
            "raw": "",
            "claims": [],
        }

    messages = [{"role": "system", "content": CLAIM_CLASSIFY_PROMPT}]
    for user, assistant in CLAIM_CLASSIFY_SHOTS:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": _claims_user_prompt(claims)})

    attempts = max(1, retries + 1)
    last_raw = ""
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            raw = chat_json(
                messages,
                model=model,
                host=host,
                temperature=0.0 if attempt == 0 else 0.2,
                num_predict=512 if attempt == 0 else 1024,
            )
            last_raw = raw
            parsed = _extract_json(raw)
            result = assemble_record(
                sample_id,
                parsed,
                audio,
                visual_lines,
                model=model,
                raw=raw,
                lexical_gate=False,
            )
            result["claims"] = claims
            return result
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                logger.warning("%s 宣称分类不合法，重试 %s/%s：%s", sample_id, attempt + 1, retries, exc)
            else:
                logger.warning("%s 宣称分类失败：%s", sample_id, exc)
    failed = _failed_record(sample_id, model, last_exc or RuntimeError("未知错误"), last_raw)
    failed["claims"] = claims
    return failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4b：宣称抽取后再少样本分类。")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", default="", help="只处理这些 sample_id，逗号分隔")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None, help="分类结果目录，默认 data/classify_claims")
    parser.add_argument("--claims-dir", type=Path, default=None, help="抽取结果目录，默认 data/claims")
    parser.add_argument("--skip-extract", action="store_true", help="复用已有 data/claims")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    asr_dir = args.data_dir / "asr"
    ocr_dir = args.data_dir / "ocr"
    claims_dir = args.claims_dir or args.data_dir / "claims"
    out_dir = args.out_dir or args.data_dir / "classify_claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
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
        logger.warning("未找到可处理样本。")
        return 0

    rows: list[dict[str, Any]] = []
    failures = 0
    n_fallback = 0
    for sample_id in ids:
        dest = out_dir / f"{sample_id}.json"
        claim_path = claims_dir / f"{sample_id}.json"
        if dest.exists() and not args.force:
            prev = _load_json(dest) or {}
            rows.append(
                {
                    "sample_id": sample_id,
                    "risk_labels": prev.get("risk_labels") or [],
                    "n_hits": prev.get("n_hits", 0),
                    "n_claims": len(prev.get("claims") or []),
                    "skipped": True,
                }
            )
            logger.info("%s skip", sample_id)
            continue

        asr = _load_json(asr_dir / f"{sample_id}.json")
        ocr = _load_json(ocr_dir / f"{sample_id}.json")
        if ocr and ocr.get("frames"):
            ocr = {**ocr, "merged": merge_ocr_frames(ocr["frames"])}

        extracted = None
        if claim_path.exists() and not args.force:
            extracted = _load_json(claim_path)
        if extracted is None:
            extracted = extract_claims(
                asr, ocr, model=args.model, host=args.host, retries=args.retries
            )
            extracted["sample_id"] = sample_id
            save_json(claim_path, extracted)
        if extracted.get("fallback"):
            n_fallback += 1

        claims = list(extracted.get("claims") or [])
        result = classify_claims(
            sample_id,
            claims,
            asr,
            ocr,
            model=args.model,
            host=args.host,
            retries=args.retries,
        )
        if str(result.get("explanation") or "").startswith("模型调用或解析失败"):
            failures += 1
        result["modality"] = "claims"
        result["experiment"] = "e2_claims"
        result["extract_fallback"] = bool(extracted.get("fallback"))
        save_json(dest, result)
        rows.append(
            {
                "sample_id": sample_id,
                "risk_labels": result["risk_labels"],
                "n_hits": result["n_hits"],
                "n_claims": len(claims),
            }
        )
        logger.info(
            "%s claims=%s %s hits=%s",
            sample_id,
            len(claims),
            ",".join(result["risk_labels"]),
            result["n_hits"],
        )

    save_json(
        args.data_dir / "classify_claims_index.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment": "e2_claims",
            "model": args.model,
            "n_samples": len(rows),
            "n_failed": failures,
            "n_extract_fallback": n_fallback,
            "prompt_note": "宣称抽取与判定的少样本只来自任务书与法规，未用测试视频原文",
            "items": rows,
        },
    )
    logger.info("写成 %s（失败 %s，抽取回退 %s）", out_dir, failures, n_fallback)
    return 1 if failures else 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
