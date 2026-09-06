# 短视频广告虚假宣传话术检测 — 进度记录

## 当前默认

本地权重在 `models/`，启动时优先读这里：

- **ASR**：FunASR `SenseVoiceSmall`
- **OCR**：PP-OCRv6 `small` 检测 + 识别
- **E2 LM**：本机 Ollama `qwen2.5:7b`（任务书示例 Qwen8B 同档，禁止 28B）

```
python -m src.pipeline --only 1,10 --force
python -m src.detect
python -m src.classify
python -m src.hybrid
python -m src.eval
python -m src.eval --detect-dir data/classify --experiment e2
python -m src.eval --detect-dir data/hybrid --experiment e3
python -m src.app
```

`--force` 只重跑未 `--skip-*` 的模态；skip-ocr 时会读已有 `data/ocr/{id}.json` 再套过滤。
E2 默认跳过已有 `data/classify/{id}.json`，加 `--force` 才重跑模型。
E3 只融合已有 E1/E2，不重跑 LLM。

## 词表纪律（e1）

`videos/` 里现有切片当作**测试集**。`src/rules.py` 只许从 `Knowledge/` 写入。
E2 少样本示例同样只来自任务书/法规，不用测试视频原文。

**禁止**根据测试视频或 P/R/F1 漏检往词表/提示词里塞本包句子。

## Done

- 环境、抽音/抽帧、ASR/OCR、全量 237 条。
- 样本信息表：`annotations/sample_info.xlsx`（编号、标题、品类、来源、日期、时长）。
- 人工标注表：`annotations/manual_annotation.xlsx`。
- **E1 规则基线**（词表冻结）：micro F1 **0.390**，macro F1 **0.219**，完全匹配 72/237。
- **E2 少样本分类**（Ollama Qwen2.5-7B，提示词示例来自 Knowledge/）
  - 风险标签 micro P/R/F1 = **0.880 / 0.362 / 0.513**；macro F1 = **0.423**；完全匹配 89/237。
- **人工复核 24 条**：`annotations/review_20.xlsx`（分层抽自分歧/漏检/稀有标签，未改词表）。
- **E3 规则∪模型**（`python -m src.hybrid`）：micro P/R/F1 = **0.897 / 0.464 / 0.612**；macro F1 **0.489**；完全匹配 106/237。诱导消费 F1 **0.885**。
- Web Demo「审言」：`python -m src.app`。
- 书面材料：`README.md`，`reports/实验报告.docx`（及 md），`reports/AI协作记录.md`，`reports/小组分工说明.md`，`reports/答辩.pptx`。姓名处为【待填】。

| 标签 | E1 F1 | E2 F1 | E3 F1 | E3 P | E3 R | support |
|---|---:|---:|---:|---:|---:|---:|
| 存在夸大功效 | 0.322 | 0.495 | **0.561** | 0.889 | 0.410 | 195 |
| 存在虚假收益承诺 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| 存在诱导消费 | 0.773 | 0.619 | **0.885** | 0.920 | 0.852 | 27 |
| 存在站外导流风险线索 | — | — | — | — | — | 0 |
| 存在其他线索 | 0.000 | **1.000** | **1.000** | 1.000 | 1.000 | 1 |

## 数据集

- `videos/`：237 条约 30 秒 mp4，已全部抽取。整包按测试集对待。
- 金标准：`annotations/`。

## 任务书收口

主要任务 1–8、实验步骤（1）–（12）、提交项（源代码、标注、报告、AI 记录、分工、答辩 PPT）均已落地。组员提交前把【待填】换成姓名，并按学院封面格式改报告首页。
