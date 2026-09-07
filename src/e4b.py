"""E4b：宣称抽取后再分类，与 E1 取并集。

不改词表、不加测试集示例。模型仍是 Ollama Qwen2.5-7B。
可选再并上 E4a 分路，写成 e4b_plus。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.claims import parse_args as claims_parse
from src.claims import run as claims_run
from src.config import ROOT
from src.eval import parse_args as eval_parse
from src.eval import run as eval_run
from src.hybrid import parse_args as hybrid_parse
from src.hybrid import run as hybrid_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("e4b")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4b：宣称抽取分类 ∪ E1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", default="", help="只处理这些 sample_id，逗号分隔")
    parser.add_argument("--force", action="store_true", help="覆盖已有抽取与分类")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-classify", action="store_true", help="只融合已有宣称分类")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plus", action="store_true", help="不跑 E4a∪宣称 的 e4b_plus")
    return parser.parse_args(argv)


def _classify_claims(args: argparse.Namespace) -> int:
    claim_args = claims_parse([])
    claim_args.data_dir = args.data_dir
    claim_args.only = args.only
    claim_args.force = args.force
    claim_args.limit = args.limit
    claim_args.retries = args.retries
    logger.info("宣称抽取 + 清单分类")
    return claims_run(claim_args)


def _eval(detect_dir: Path, experiment: str) -> int:
    eval_args = eval_parse([])
    eval_args.detect_dir = detect_dir
    eval_args.out_dir = detect_dir.parent / "eval"
    eval_args.experiment = experiment
    return eval_run(eval_args)


def run(args: argparse.Namespace) -> int:
    if not args.skip_classify:
        code = _classify_claims(args)
        if code:
            logger.warning("宣称分类有失败条目，继续融合已写出的结果")

    hybrid_args = hybrid_parse([])
    hybrid_args.data_dir = args.data_dir
    hybrid_args.detect_dir = args.data_dir / "detect"
    hybrid_args.classify_dir = [args.data_dir / "classify_claims"]
    hybrid_args.engine_name = ["model_claims"]
    hybrid_args.out_dir = args.data_dir / "e4b"
    hybrid_args.experiment = "e4b"
    hybrid_args.note = "E4b：E1 ∪ 宣称抽取后分类；不改词表，不把测试集句子写入提示词"
    hybrid_args.index_name = "e4b_index.json"
    code = hybrid_run(hybrid_args)
    if code:
        return code

    if not args.skip_eval:
        _eval(args.data_dir / "classify_claims", "e2_claims")
        code = _eval(args.data_dir / "e4b", "e4b")
        if code:
            return code

    plus_ok = False
    audio_dir = args.data_dir / "classify_audio"
    visual_dir = args.data_dir / "classify_visual"
    if not args.skip_plus and audio_dir.exists() and visual_dir.exists():
        plus_args = hybrid_parse([])
        plus_args.data_dir = args.data_dir
        plus_args.detect_dir = args.data_dir / "detect"
        plus_args.classify_dir = [audio_dir, visual_dir, args.data_dir / "classify_claims"]
        plus_args.engine_name = ["model_audio", "model_visual", "model_claims"]
        plus_args.out_dir = args.data_dir / "e4b_plus"
        plus_args.experiment = "e4b_plus"
        plus_args.note = "E4b+：E1 ∪ 口播 ∪ 画面 ∪ 宣称；不改词表"
        plus_args.index_name = "e4b_plus_index.json"
        plus_code = hybrid_run(plus_args)
        if plus_code:
            logger.warning("e4b_plus 融合失败")
        elif not args.skip_eval:
            _eval(args.data_dir / "e4b_plus", "e4b_plus")
            plus_ok = True
        else:
            plus_ok = True

    if not args.skip_eval:
        _print_vs(args.data_dir / "eval", plus_ok)
    return 0


def _print_vs(eval_dir: Path, plus_ok: bool) -> None:
    names = ["e3", "e4a", "e2_claims", "e4b"]
    if plus_ok:
        names.append("e4b_plus")
    reports: dict[str, dict] = {}
    for name in names:
        path = eval_dir / f"{name}_metrics.json"
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    if "e4b" not in reports:
        return
    print("--- vs E3 / E4a ---")
    for title, key in (("micro_f1", "micro_risk"), ("macro_f1", "macro_risk")):
        bits = []
        for name in names:
            if name in reports:
                bits.append(f"{name}={reports[name].get(key, {}).get('f1')}")
        print(f"{title}: " + " ".join(bits))
    bits = []
    for name in names:
        if name in reports:
            bits.append(f"{name}={reports[name].get('n_exact')}")
    print("exact_match: " + " ".join(bits))
    for lab in ("存在夸大功效", "存在虚假收益承诺", "存在诱导消费"):
        parts = [lab]
        for name in names:
            if name not in reports:
                continue
            row = reports[name].get("per_label", {}).get(lab, {})
            parts.append(f"{name}={row.get('precision')}/{row.get('recall')}/{row.get('f1')}")
        print(" ".join(parts))


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
