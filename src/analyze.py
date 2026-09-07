"""单条视频审核：抽取 ASR/OCR，再按规则 / 模型 / 融合给出结构化结果。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from src.asr import build_asr_engine, save_json
from src.classify import classify_record
from src.config import ROOT, Settings, prepare_hf_env
from src.detect import detect_record
from src.llm import DEFAULT_HOST, DEFAULT_MODEL, LLMError, ensure_model, list_ollama_models
from src.media import probe_duration
from src.merge import merge_ocr_frames
from src.ocr import PaddleOCREngine
from src.pipeline import process_one
from src.rules import LABEL_NORMAL, RISK_LABELS, finalize_labels

logger = logging.getLogger("analyze")

ProgressFn = Callable[[str, str, dict[str, Any]], None]

MODES = {
    "rules": {
        "id": "rules",
        "name": "规则基线",
        "short": "词表命中",
        "needs_llm": False,
        "blurb": "按任务书与《广告法》词表精确匹配，证据可回原文、依据可复核。",
    },
    "model": {
        "id": "model",
        "name": "模型分类",
        "short": "少样本 LM",
        "needs_llm": True,
        "blurb": "本地 Qwen2.5-7B 少样本分类，能抓住词表未覆盖的功效断言。",
    },
    "hybrid": {
        "id": "hybrid",
        "name": "规则 + 模型",
        "short": "互补融合",
        "needs_llm": True,
        "blurb": "取并集：规则稳住促销误导，模型补上夸大功效。",
    },
    "e4": {
        "id": "e4",
        "name": "分轨识别",
        "short": "口播 / 画面",
        "needs_llm": True,
        "blurb": "口播与画面分开分类，再与规则取并，避免合路时漏掉较弱一侧。",
    },
}

_LOCK = threading.Lock()
_ASR: Any = None
_OCR: PaddleOCREngine | None = None


def _emit(on_progress: ProgressFn | None, stage: str, message: str, **extra: Any) -> None:
    if on_progress:
        on_progress(stage, message, extra)


def ollama_status(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    try:
        names = list_ollama_models(host)
        wanted = {model, f"{model}:latest"}
        ready = any(
            n in wanted or n.startswith(f"{model}:") or n.startswith(f"{model}-") for n in names
        )
        return {"ok": True, "ready": ready, "model": model, "installed": names}
    except LLMError as exc:
        return {"ok": False, "ready": False, "model": model, "installed": [], "error": str(exc)}


def engines_loaded() -> dict[str, bool]:
    return {"asr": _ASR is not None, "ocr": _OCR is not None}


def _ensure_extract_engines(settings: Settings, on_progress: ProgressFn | None) -> tuple[Any, PaddleOCREngine]:
    global _ASR, _OCR
    with _LOCK:
        if _ASR is None:
            _emit(on_progress, "load_asr", "正在加载口播识别模型 SenseVoice…")
            prepare_hf_env()
            _ASR = build_asr_engine(
                backend=settings.asr_backend,
                model_size=settings.asr_model,
                device=settings.asr_device,
                compute_type=settings.asr_compute_type,
                language=settings.asr_language,
                models_dir=settings.models_dir,
            )
        if _OCR is None:
            _emit(on_progress, "load_ocr", "正在加载画面识别模型 PP-OCRv6…")
            _OCR = PaddleOCREngine(
                lang=settings.ocr_lang,
                min_score=settings.min_ocr_score,
                models_dir=settings.models_dir,
                det_dir=settings.ocr_det_dir,
                rec_dir=settings.ocr_rec_dir,
                ocr_size=settings.ocr_size,
            )
        return _ASR, _OCR


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _extract(
    video_path: Path,
    sample_id: str,
    settings: Settings,
    *,
    force: bool,
    on_progress: ProgressFn | None,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    settings.ensure_dirs()
    asr_path = settings.asr_dir / f"{sample_id}.json"
    ocr_path = settings.ocr_dir / f"{sample_id}.json"
    duration = probe_duration(video_path)

    asr = None if force else _load_json(asr_path)
    ocr = None if force else _load_json(ocr_path)
    if asr is not None and ocr is not None:
        _emit(on_progress, "cache", "复用已抽取的口播与画面文本")
        if ocr.get("frames") and not ocr.get("merged"):
            ocr = {**ocr, "merged": merge_ocr_frames(ocr["frames"])}
        return asr, ocr, duration

    asr_engine, ocr_engine = _ensure_extract_engines(settings, on_progress)
    meta = {"sample_id": sample_id}
    # force 时只重跑缺失的一侧；两侧都缺则整段抽取。
    local = Settings(
        videos_dir=settings.videos_dir,
        data_dir=settings.data_dir,
        models_dir=settings.models_dir,
        frame_interval=settings.frame_interval,
        frame_change_threshold=settings.frame_change_threshold,
        asr_backend=settings.asr_backend,
        asr_model=settings.asr_model,
        ocr_size=settings.ocr_size,
        skip_asr=asr is not None and not force,
        skip_ocr=ocr is not None and not force,
        force=force,
        save_frames=False,
    )
    _emit(on_progress, "extract", "抽取口播转写与画面文字，首次会较慢…")
    merged = process_one(video_path, local, asr_engine, ocr_engine, meta)
    asr = _load_json(asr_path) or {"text": merged.get("audio_text") or "", "segments": []}
    ocr = _load_json(ocr_path) or {"frames": [], "merged": []}
    if ocr.get("frames") and not ocr.get("merged"):
        ocr = {**ocr, "merged": merge_ocr_frames(ocr["frames"])}
    save_json(settings.merged_dir / f"{sample_id}.json", merged)
    return asr, ocr, duration


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _hit_rows(result: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in result.get("hits") or []:
        if engine == "rules" and hit.get("hedged"):
            continue
        if engine != "rules" and hit.get("grounded") is False:
            continue
        label = str(hit.get("label") or "")
        if label not in RISK_LABELS:
            continue
        pos = hit.get("evidence_position") or {}
        span = pos.get("span")
        rows.append(
            {
                "label": label,
                "evidence": str(hit.get("evidence") or ""),
                "matched": str(hit.get("matched") or hit.get("evidence") or ""),
                "source": pos.get("source") or "unknown",
                "time": _to_float(pos.get("time")),
                "span": [int(x) for x in span] if isinstance(span, (list, tuple)) else None,
                "rule_basis": str(hit.get("rule_basis") or ""),
                "hedged": bool(hit.get("hedged")),
                "engine": engine,
            }
        )
    if rows:
        return rows
    evidence = result.get("evidence") or []
    positions = result.get("evidence_position") or []
    labels = [x for x in (result.get("risk_labels") or []) if x in RISK_LABELS]
    for i, text in enumerate(evidence):
        pos = positions[i] if i < len(positions) and isinstance(positions[i], dict) else {}
        span = pos.get("span")
        rows.append(
            {
                "label": labels[0] if labels else LABEL_NORMAL,
                "evidence": str(text),
                "matched": str(text),
                "source": pos.get("source") or "unknown",
                "time": _to_float(pos.get("time")),
                "span": [int(x) for x in span] if isinstance(span, (list, tuple)) else None,
                "rule_basis": (result.get("rule_basis") or [""])[0] if result.get("rule_basis") else "",
                "hedged": False,
                "engine": engine,
            }
        )
    return rows


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for hit in hits:
        key = (hit["label"], hit["source"], (hit.get("matched") or hit.get("evidence") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


def _verdict(labels: list[str]) -> tuple[str, str]:
    risks = [x for x in labels if x in RISK_LABELS]
    if risks:
        return "risk", "涉及虚假宣传"
    return "normal", "未见虚假宣传话术"


_ENGINE_CN = {
    "rules": "规则",
    "model": "模型",
    "model_audio": "口播模型",
    "model_visual": "画面模型",
    "model_claims": "宣称模型",
}


def _fuse_many(named: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """多路风险标签并集。无风险时输出正常，不发出无法判断。"""
    if not named:
        return {
            "risk_labels": [LABEL_NORMAL],
            "hits": [],
            "rule_basis": [],
            "explanation": "没有可融合的预测。",
            "n_hits": 0,
            "n_hedged": 0,
        }

    risks_map: dict[str, list[str]] = {}
    for name, rec in named:
        labs = finalize_labels(list(rec.get("risk_labels") or []))
        risks_map[name] = [x for x in RISK_LABELS if x in labs]

    fused = [x for x in RISK_LABELS if any(x in risks for risks in risks_map.values())]
    labels = fused if fused else [LABEL_NORMAL]

    hits = _dedupe_hits([h for name, rec in named for h in _hit_rows(rec, name)])
    hits = [h for h in hits if h["label"] in fused] if fused else []

    bits: list[str] = []
    for name, risks in risks_map.items():
        if risks:
            bits.append(f"{_ENGINE_CN.get(name, name)}命中：" + "、".join(risks))

    extras: list[str] = []
    for name, risks in risks_map.items():
        others = {x for other, rs in risks_map.items() if other != name for x in rs}
        only = [x for x in risks if x not in others]
        if only:
            extras.append(f"{_ENGINE_CN.get(name, name)}补上「" + "、".join(only) + "」")
    if extras:
        bits.append("；".join(extras) + "。")

    preferred = ""
    if labels != [LABEL_NORMAL]:
        for name, rec in reversed(named):
            if name == "rules":
                continue
            exp = str(rec.get("explanation") or "").strip()
            if exp:
                preferred = exp
                break
    if not preferred:
        for _, rec in named:
            exp = str(rec.get("explanation") or "").strip()
            if exp:
                preferred = exp
                break
    if preferred:
        bits.append(preferred)
    if not bits:
        bits.append("各路均未给出风险标签。")

    basis: list[str] = []
    for _, rec in named:
        for item in rec.get("rule_basis") or []:
            text = str(item).strip()
            if text and text not in basis:
                basis.append(text)

    n_hedged = 0
    for name, rec in named:
        if name == "rules":
            n_hedged = int(rec.get("n_hedged") or 0)
            break

    return {
        "risk_labels": labels,
        "hits": hits,
        "rule_basis": basis,
        "explanation": " ".join(bits),
        "n_hits": len(hits),
        "n_hedged": n_hedged,
    }


def _fuse(e1: dict[str, Any], e2: dict[str, Any]) -> dict[str, Any]:
    return _fuse_many([("rules", e1), ("model", e2)])


def _pack(
    *,
    sample_id: str,
    video_name: str,
    duration: float,
    mode: str,
    result: dict[str, Any],
    asr: dict[str, Any],
    ocr: dict[str, Any],
    engines: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = finalize_labels(list(result.get("risk_labels") or []))
    verdict, verdict_text = _verdict(labels)
    hits = result.get("hits")
    if not isinstance(hits, list) or (hits and "engine" not in hits[0]):
        hits = _hit_rows(result, "rules" if mode == "rules" else "model")
    visual = ocr.get("merged") or merge_ocr_frames(ocr.get("frames") or [])
    return {
        "sample_id": sample_id,
        "video_name": video_name,
        "duration_sec": round(float(duration or 0.0), 3),
        "mode": mode,
        "mode_meta": MODES[mode],
        "verdict": verdict,
        "verdict_text": verdict_text,
        "involves_false_ad": verdict == "risk",
        "risk_labels": labels,
        "explanation": str(result.get("explanation") or ""),
        "rule_basis": list(result.get("rule_basis") or []),
        "evidence": hits,
        "n_hits": len(hits),
        "asr": {
            "text": str(asr.get("text") or ""),
            "segments": asr.get("segments") or [],
        },
        "ocr": {
            "merged": [
                {
                    "text": str(line.get("text") or ""),
                    "time": _to_float(line.get("time")),
                    "score": _to_float(line.get("score")),
                }
                for line in visual
                if str(line.get("text") or "").strip()
            ]
        },
        "engines": engines or {},
        "model": result.get("model"),
    }


def analyze_video(
    video_path: Path,
    sample_id: str,
    mode: str,
    *,
    force: bool = False,
    on_progress: ProgressFn | None = None,
    settings: Settings | None = None,
    llm_model: str = DEFAULT_MODEL,
    llm_host: str = DEFAULT_HOST,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"未知分析模式 {mode!r}，可选：{', '.join(MODES)}")
    settings = settings or Settings(videos_dir=ROOT / "videos", data_dir=ROOT / "data")
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频：{video_path}")

    asr, ocr, duration = _extract(
        video_path, sample_id, settings, force=force, on_progress=on_progress
    )
    if ocr.get("frames"):
        ocr = {**ocr, "merged": merge_ocr_frames(ocr["frames"])}

    meta = MODES[mode]
    engines: dict[str, Any] = {}

    if meta["needs_llm"]:
        _emit(on_progress, "llm", f"检查本地模型 {llm_model}…")
        ensure_model(llm_model, llm_host)

    if mode == "rules":
        _emit(on_progress, "detect", "规则词表审核中…")
        result = detect_record(sample_id, asr, ocr)
        save_json(settings.detect_dir / f"{sample_id}.json", result)
        engines["rules"] = {"risk_labels": result["risk_labels"], "n_hits": result.get("n_hits", 0)}
        packed_hits = _hit_rows(result, "rules")
        result = {**result, "hits": packed_hits, "n_hits": len(packed_hits)}
        return _pack(
            sample_id=sample_id,
            video_name=video_path.name,
            duration=duration,
            mode=mode,
            result=result,
            asr=asr,
            ocr=ocr,
            engines=engines,
        )

    if mode == "model":
        _emit(on_progress, "classify", "模型少样本分类中…")
        result = classify_record(sample_id, asr, ocr, model=llm_model, host=llm_host)
        save_json(settings.classify_dir / f"{sample_id}.json", result)
        engines["model"] = {"risk_labels": result["risk_labels"], "n_hits": result.get("n_hits", 0)}
        packed_hits = _hit_rows(result, "model")
        result = {**result, "hits": packed_hits, "n_hits": len(packed_hits)}
        return _pack(
            sample_id=sample_id,
            video_name=video_path.name,
            duration=duration,
            mode=mode,
            result=result,
            asr=asr,
            ocr=ocr,
            engines=engines,
        )

    _emit(on_progress, "detect", "规则词表审核中…")
    e1 = detect_record(sample_id, asr, ocr)
    save_json(settings.detect_dir / f"{sample_id}.json", e1)

    if mode == "e4":
        _emit(on_progress, "classify", "口播分轨识别中…")
        e_audio = classify_record(
            sample_id, asr, {"merged": [], "frames": []}, model=llm_model, host=llm_host
        )
        audio_dir = settings.data_dir / "classify_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        save_json(audio_dir / f"{sample_id}.json", e_audio)
        _emit(on_progress, "classify", "画面分轨识别中…")
        e_visual = classify_record(
            sample_id, {"text": "", "segments": []}, ocr, model=llm_model, host=llm_host
        )
        visual_dir = settings.data_dir / "classify_visual"
        visual_dir.mkdir(parents=True, exist_ok=True)
        save_json(visual_dir / f"{sample_id}.json", e_visual)
        engines = {
            "rules": {"risk_labels": e1["risk_labels"], "n_hits": e1.get("n_hits", 0)},
            "model_audio": {"risk_labels": e_audio["risk_labels"], "n_hits": e_audio.get("n_hits", 0)},
            "model_visual": {"risk_labels": e_visual["risk_labels"], "n_hits": e_visual.get("n_hits", 0)},
        }
        fused = _fuse_many(
            [("rules", e1), ("model_audio", e_audio), ("model_visual", e_visual)]
        )
        _emit(on_progress, "fuse", "融合规则与分轨结果…")
        return _pack(
            sample_id=sample_id,
            video_name=video_path.name,
            duration=duration,
            mode=mode,
            result=fused,
            asr=asr,
            ocr=ocr,
            engines=engines,
        )

    _emit(on_progress, "classify", "模型少样本分类中…")
    e2 = classify_record(sample_id, asr, ocr, model=llm_model, host=llm_host)
    save_json(settings.classify_dir / f"{sample_id}.json", e2)
    engines = {
        "rules": {"risk_labels": e1["risk_labels"], "n_hits": e1.get("n_hits", 0)},
        "model": {"risk_labels": e2["risk_labels"], "n_hits": e2.get("n_hits", 0)},
    }
    fused = _fuse(e1, e2)
    _emit(on_progress, "fuse", "融合规则与模型结果…")
    return _pack(
        sample_id=sample_id,
        video_name=video_path.name,
        duration=duration,
        mode=mode,
        result=fused,
        asr=asr,
        ocr=ocr,
        engines=engines,
    )
