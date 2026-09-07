"""E4 分轨识别：口播 / 画面分路少样本分类，再与 E1 取并集。

不改词表、不加测试集示例。模型仍是 Ollama Qwen2.5-7B。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.classify import parse_args as classify_parse
from src.classify import run as classify_run
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
logger = logging.getLogger("e4a")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E4 分轨识别：ASR-only ∪ OCR-only ∪ E1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", default="", help="只处理这些 sample_id，逗号分隔")
    parser.add_argument("--force", action="store_true", help="覆盖已有分路分类")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-classify", action="store_true", help="只融合已有分路结果")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def _classify(args: argparse.Namespace, modality: str) -> int:
    classify_args = classify_parse([])
    classify_args.data_dir = args.data_dir
    classify_args.only = args.only
    classify_args.force = args.force
    classify_args.limit = args.limit
    classify_args.retries = args.retries
    classify_args.modality = modality
    classify_args.from_raw = False
    classify_args.out_dir = None
    logger.info("分路分类 modality=%s", modality)
    return classify_run(classify_args)


def run(args: argparse.Namespace) -> int:
    if not args.skip_classify:
        for modality in ("audio", "visual"):
            code = _classify(args, modality)
            if code:
                logger.warning("分路分类有失败条目 modality=%s，继续融合已写出的结果", modality)

    hybrid_args = hybrid_parse([])
    hybrid_args.data_dir = args.data_dir
    hybrid_args.detect_dir = args.data_dir / "detect"
    hybrid_args.classify_dir = [
        args.data_dir / "classify_audio",
        args.data_dir / "classify_visual",
    ]
    hybrid_args.engine_name = ["model_audio", "model_visual"]
    hybrid_args.out_dir = args.data_dir / "e4a"
    hybrid_args.experiment = "e4a"
    hybrid_args.note = "E4 分轨识别：E1 ∪ 口播模型 ∪ 画面模型；不改词表，不把测试集句子写入提示词"
    hybrid_args.index_name = "e4a_index.json"
    code = hybrid_run(hybrid_args)
    if code:
        return code

    if args.skip_eval:
        return 0

    eval_args = eval_parse([])
    eval_args.detect_dir = args.data_dir / "e4a"
    eval_args.out_dir = args.data_dir / "eval"
    eval_args.experiment = "e4a"
    code = eval_run(eval_args)
    _print_vs_e3(args.data_dir / "eval")
    return code


def _print_vs_e3(eval_dir: Path) -> None:
    e3_path = eval_dir / "e3_metrics.json"
    e4_path = eval_dir / "e4a_metrics.json"
    if not e3_path.exists() or not e4_path.exists():
        return
    e3 = json.loads(e3_path.read_text(encoding="utf-8"))
    e4 = json.loads(e4_path.read_text(encoding="utf-8"))
    print("--- vs E3 ---")
    for title, key in (("micro_f1", "micro_risk"), ("macro_f1", "macro_risk")):
        print(f"{title}: e3={e3.get(key, {}).get('f1')} e4a={e4.get(key, {}).get('f1')}")
    print(f"exact_match: e3={e3.get('n_exact')} e4a={e4.get('n_exact')}")
    for lab in ("存在夸大功效", "存在虚假收益承诺", "存在诱导消费"):
        a = e3.get("per_label", {}).get(lab, {})
        b = e4.get("per_label", {}).get(lab, {})
        print(
            f"{lab}: e3 P/R/F1={a.get('precision')}/{a.get('recall')}/{a.get('f1')} "
            f"e4a={b.get('precision')}/{b.get('recall')}/{b.get('f1')}"
        )


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
