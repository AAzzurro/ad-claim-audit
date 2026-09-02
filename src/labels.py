from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_DROP_COLS = {"原始数据", "视频地址"}
_ID_COLS = (
    "sample_id",
    "sampleid",
    "id",
    "编号",
    "样本编号",
    "视频编号",
)
_FILE_COLS = (
    "video_name",
    "filename",
    "file",
    "path",
    "video",
    "视频文件",
    "文件名",
    "视频名",
    "文件",
)


def discover_labels(search_dirs: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for folder in search_dirs:
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.suffix.lower() in {".csv", ".xlsx", ".xls"} and path.is_file():
                if path.name.startswith("~$"):
                    continue
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.suffix.lower() != ".xlsx", p.name))
    return candidates[0]


def load_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _norm(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text


def index_labels(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """用 sample_id / 文件名（含无后缀）做查找。"""
    mapping: dict[str, dict[str, Any]] = {}
    columns = {str(c).strip(): c for c in df.columns}
    lower = {k.lower(): v for k, v in columns.items()}

    id_col = next((lower[c] for c in _ID_COLS if c in lower), None)
    file_col = next((lower[c] for c in _FILE_COLS if c in lower), None)

    for _, row in df.iterrows():
        record = {
            str(k): _norm(v)
            for k, v in row.items()
            if str(k) not in _DROP_COLS
        }
        keys = []
        if id_col is not None:
            keys.append(_norm(row[id_col]))
        if file_col is not None:
            raw = _norm(row[file_col])
            keys.append(raw)
            keys.append(Path(raw).name)
            keys.append(Path(raw).stem)
        for key in keys:
            if key:
                mapping[key] = record
                mapping[key.lower()] = record
    logger.info("标签表 %s 行，建立 %s 条索引", len(df), len(mapping))
    return mapping


def lookup_meta(index: dict[str, dict[str, Any]], video_path: Path) -> dict[str, Any]:
    for key in (video_path.stem, video_path.name, video_path.stem.lower(), video_path.name.lower()):
        if key in index:
            return index[key]
    return {}
