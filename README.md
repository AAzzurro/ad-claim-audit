# 短视频广告虚假宣传话术检测

以约 30 秒直播/短视频广告切片为输入，完成口播转写（ASR）、画面文字识别（OCR）、规则基线（E1）、少样本语言模型分类（E2）和规则+模型融合（E3），输出风险标签、原文证据、位置、法规依据和解释。Web Demo 名称为「审言」。

任务书见 `Knowledge/短视频广告虚假宣传话术检测.pdf`。测试集约 237 条，整包按测试集对待，**不根据本包句子回写词表或少样本示例**。

## 环境

```bash
./setup_env.sh
source .venv/bin/activate
```

默认本地权重在 `models/`：

- ASR：FunASR SenseVoiceSmall
- OCR：PP-OCRv6 small
- E2：本机 Ollama `qwen2.5:7b`（任务书要求小规模 LM，禁止 28B）

需本机已安装 [ffmpeg](https://ffmpeg.org/) 与 [Ollama](https://ollama.com/)，并执行过 `ollama pull qwen2.5:7b`。

无 GPU / 无法重跑 ASR/OCR 时，可直接使用仓库中已抽取的 `data/asr/`、`data/ocr/` 做检测与评测。

## 流水线

```bash
python -m src.pipeline --only 1,10 --force   # 抽取口播与画面
python -m src.detect                         # E1 规则基线
python -m src.classify                       # E2 少样本分类（需 Ollama）
python -m src.hybrid                         # E3 融合已有 E1/E2，不重跑模型
python -m src.eval                           # 评 E1
python -m src.eval --detect-dir data/classify --experiment e2
python -m src.eval --detect-dir data/hybrid --experiment e3
python -m src.app                            # Demo http://127.0.0.1:7860
```

`--force` 只重跑未 `--skip-*` 的模态。E2 默认跳过已有 `data/classify/{id}.json`。

## 数据与标注

| 路径 | 说明 |
|---|---|
| `annotations/sample_info.xlsx` | 样本信息表（编号、标题、品类、来源、日期、时长） |
| `annotations/manual_annotation.xlsx` | 人工标注表与证据明细 |
| `annotations/review_20.xlsx` | ≥20 条预测人工复核表 |
| `data/asr` `data/ocr` `data/merged` | 已抽取文本 |
| `data/detect` `data/classify` `data/hybrid` | E1 / E2 / E3 预测 |
| `data/eval` | P/R/F1 与分歧表 |
| `videos/` | 原视频（体积大，可不提交；可用转写文本复现） |
| `reports/` | 实验报告、AI 协作记录、分工说明、答辩 PPT |

## 标签

多标签：`存在夸大功效`、`存在虚假收益承诺`、`存在诱导消费`、`存在站外导流风险线索`、`存在其他线索`；互斥特殊值：`正常`、`无法判断`。

判断只依据本条切片中可见、可听到的内容，不推测商品真实效果。
