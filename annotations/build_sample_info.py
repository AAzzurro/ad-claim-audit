"""从抽取索引与人工标注表生成任务书要求的样本信息表。"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "annotations"
INDEX_PATH = ROOT / "data" / "extract_index.json"
GOLD_PATH = OUT_DIR / "manual_annotation.json"
SOURCE = "教师提供公开直播/短视频广告切片"


def _clean_category(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return "；".join(str(x).strip() for x in parsed if str(x).strip())
        except (SyntaxError, ValueError):
            pass
    return text.strip("[]'\" ")


def _title(sample_id: str, category: str, copy_text: str) -> str:
    cat = category.split("；", 1)[0].strip()
    if "-" in cat:
        cat = cat.split("-", 1)[1].strip()
    head = (copy_text or "").split("，", 1)[0].split("。", 1)[0].strip()
    if len(head) > 28:
        head = head[:28] + "…"
    if cat and head:
        return f"切片{sample_id} · {cat} · {head}"
    if cat:
        return f"切片{sample_id} · {cat}"
    return f"切片{sample_id}"


def main() -> None:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    gold_rows = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    gold = {str(r["sample_id"]): r for r in gold_rows}
    duration = {
        str(it["sample_id"]): it.get("duration_sec")
        for it in index.get("items") or []
    }

    rows = []
    for sid, rec in gold.items():
        category = _clean_category(str(rec.get("商品类别") or ""))
        copy_text = str(rec.get("文案") or "")
        rows.append(
            {
                "sample_id": sid,
                "视频文件名": rec.get("video_name") or f"{sid}.mp4",
                "视频标题": _title(sid, category, copy_text),
                "商品/服务类别": category,
                "来源": SOURCE,
                "采集日期": rec.get("标注日期") or "2026-09-04",
                "视频时长_秒": duration.get(sid),
                "文案摘要": copy_text,
            }
        )

    df = pd.DataFrame(rows)
    df["sid_n"] = pd.to_numeric(df["sample_id"], errors="coerce")
    df = df.sort_values("sid_n").drop(columns=["sid_n"])

    csv_path = OUT_DIR / "sample_info.csv"
    xlsx_path = OUT_DIR / "sample_info.xlsx"
    json_path = OUT_DIR / "sample_info.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"samples={len(df)} csv={csv_path} xlsx={xlsx_path}")
    print(df["商品/服务类别"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
