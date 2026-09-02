#!/usr/bin/env zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "找不到 python3"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "创建虚拟环境 .venv （$PY）"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip setuptools wheel

# 国内 PyPI 镜像；失败时脚本仍会继续用官方源
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "安装通用依赖"
python -m pip install -i "$PIP_INDEX" numpy "opencv-python-headless>=4.8" imageio-ffmpeg tqdm pandas openpyxl

echo "安装 faster-whisper（中文 ASR）"
python -m pip install -i "$PIP_INDEX" "faster-whisper>=1.1" "huggingface_hub>=0.23,<1"

echo "安装 PaddlePaddle CPU（Apple Silicon 官方源）"
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

echo "安装 PaddleOCR"
python -m pip install -i "$PIP_INDEX" "paddleocr>=3.0"

echo "写入 requirements-lock.txt"
python -m pip freeze > requirements-lock.txt

echo "检查导入"
python -m src.check_env

echo
echo "完成。之后运行："
echo "  source .venv/bin/activate"
echo "  python -m src.pipeline          # 处理 videos/ 下全部 mp4"
echo "  python -m src.pipeline --limit 2   # 先拿 2 条试跑"
