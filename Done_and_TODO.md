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
python -m src.eval
python -m src.eval --detect-dir data/classify --experiment e2
```

`--force` 只重跑未 `--skip-*` 的模态；skip-ocr 时会读已有 `data/ocr/{id}.json` 再套过滤。
E2 默认跳过已有 `data/classify/{id}.json`，加 `--force` 才重跑模型。

## 词表纪律（e1）

`videos/` 里现有切片当作**测试集**。`src/rules.py` 只许从 `Knowledge/` 写入。
E2 少样本示例同样只来自任务书/法规，不用测试视频原文。

**禁止**根据测试视频或 P/R/F1 漏检往词表/提示词里塞本包句子。

## Done

- 环境、抽音/抽帧、ASR/OCR、全量 237 条。
- 人工标注表：`annotations/manual_annotation.xlsx`。
- **E1 规则基线**（词表冻结）：micro F1 **0.390**，macro F1 **0.219**，完全匹配 72/237。
- **2026-09-04 E2 少样本分类**（Ollama Qwen2.5-7B，提示词示例来自 Knowledge/）
  - `python -m src.classify` → `data/classify/{id}.json`
  - 字段与 E1 对齐：`risk_labels` / `evidence` / `evidence_position` / `rule_basis` / `explanation`
  - 证据短句需能在原文落地；诱导消费/站外导流/其他线索还要能对上任务书定义，避免把领券、拍链接当成风险
  - 237 条。风险标签 micro P/R/F1 = **0.880 / 0.362 / 0.513**；macro F1 = **0.423**；完全匹配 89/237（37.6%）

| 标签 | E1 F1 | E2 F1 | E2 P | E2 R | support |
|---|---:|---:|---:|---:|---:|
| 存在夸大功效 | 0.322 | **0.495** | 0.882 | 0.344 | 195 |
| 存在虚假收益承诺 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| 存在诱导消费 | **0.773** | 0.619 | 0.867 | 0.482 | 27 |
| 存在站外导流风险线索 | — | — | — | — | 0 |
| 存在其他线索 | 0.000 | **1.000** | 1.000 | 1.000 | 1 |

  - 相对 E1：功效召回上升（能抓住「断根」「治疗」等词表外断言），诱导消费召回下降（E1 的「最低价」更稳）。两者互补。
  - 典型问题：192 加盟收益仍漏；10 画面「天花板」漏、只打中「活动就一天」；162 标签对了但证据取了「可达0.4」而非「治疗骨关节炎」。

## 数据集

- `videos/`：237 条约 30 秒 mp4，已全部抽取。整包按测试集对待。
- 金标准：`annotations/`。

## 下一步（只做这一件）

按任务书做 **人工复核至少 20 条**预测结果：从 E1/E2 分歧和漏检里抽样，填复核表（样本编号、人工标签、模型标签、证据是否正确、是否需要修改、误判原因）。不要此时改词表或提示词。
