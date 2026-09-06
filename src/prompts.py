"""E2 少样本提示词。示例只来自任务书/法规，不用测试视频原文。"""

from __future__ import annotations

from src.rules import (
    LABEL_EFFICACY,
    LABEL_INDUCE,
    LABEL_NORMAL,
    LABEL_OFFSITE,
    LABEL_OTHER,
    LABEL_UNKNOWN,
    LABEL_YIELD,
    RISK_LABELS,
)

ALLOWED_LABELS = (LABEL_EFFICACY, LABEL_YIELD, LABEL_INDUCE, LABEL_OFFSITE, LABEL_OTHER, LABEL_NORMAL, LABEL_UNKNOWN)

SYSTEM_PROMPT = f"""你是短视频广告审核助手。只依据【口播 ASR】和【画面 OCR】原文，不推测商品真假。
风险可多选；「{LABEL_NORMAL}」「{LABEL_UNKNOWN}」不可与风险同时出现。先看画面叠字。

- {LABEL_EFFICACY}：功效/疗效断言；疾病治疗或预防；绝对化（国家级、最高级、最佳、第一、唯一、首个、首创、首款、天花板）。「没有天花板」或把天花板当预算上限则不标。
- {LABEL_YIELD}：稳赚、保本、无风险、日入、包过。
- {LABEL_INDUCE}：最后几单、仅此一次、错过不再、活动就一天、全网/全年/历史最低价。
- {LABEL_OFFSITE}：扫码加群、加微信、私信下单。拍链接不是导流。
- {LABEL_OTHER}：仅迷信好运/转运。领券、催单、粉丝反馈不是。
- {LABEL_NORMAL}：普通介绍、常规优惠；有条件描述（可能、有助于、效果因人而异）。
- {LABEL_UNKNOWN}：原文过短或乱码。

「最高领取N元」「券后价」「包邮」不是最高级。商品标价不是收益承诺。导向、骂人、外貌贬低不属本任务。
医疗/药品/保健食品/金融从严。
每项风险必须带原文子串证据。只输出一个 JSON：
{{"risk_labels":["{LABEL_EFFICACY}"],"evidence":[{{"label":"{LABEL_EFFICACY}","text":"原文子串","source":"audio或visual"}}],"rule_basis":["条款"],"explanation":"一句"}}
"""

# 少样本：任务书背景示例 + 广告法条文，不用 videos/ 里的句子。
FEW_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "【口播 ASR】\n这款是行业天花板，能根治，三天见效。\n【画面 OCR】\n速效配方",
        (
            '{"risk_labels":["存在夸大功效"],'
            '"evidence":['
            '{"label":"存在夸大功效","text":"行业天花板","source":"audio"},'
            '{"label":"存在夸大功效","text":"根治","source":"audio"}'
            "],"
            '"rule_basis":["《广告法》第九、十六条；任务书示例「行业天花板」「根治」"],'
            '"explanation":"绝对化加功效断言。"}'
        ),
    ),
    (
        "【口播 ASR】\n这款保健食品国内首创，可以治疗失眠。\n【画面 OCR】\n",
        (
            '{"risk_labels":["存在夸大功效"],'
            '"evidence":['
            '{"label":"存在夸大功效","text":"国内首创","source":"audio"},'
            '{"label":"存在夸大功效","text":"治疗失眠","source":"audio"}'
            "],"
            '"rule_basis":["《广告法》第九条「等」与第一同类；第十七条、十八条不得涉及疾病治疗"],'
            '"explanation":"首创为绝对化，治疗疾病为医疗用语。"}'
        ),
    ),
    (
        "【口播 ASR】\n最后几单了，全网最低价，错过不再。\n【画面 OCR】\n全年最低价",
        (
            '{"risk_labels":["存在诱导消费"],'
            '"evidence":['
            '{"label":"存在诱导消费","text":"最后几单","source":"audio"},'
            '{"label":"存在诱导消费","text":"全网最低价","source":"audio"}'
            "],"
            '"rule_basis":["《直播电商监督管理办法》第三十二条"],'
            '"explanation":"限时稀缺加虚构最低价。"}'
        ),
    ),
    (
        "【口播 ASR】\n可能有助于改善肤感，效果因人而异。\n【画面 OCR】\n最高可领优惠 券后价 2号链接拍下",
        (
            '{"risk_labels":["正常"],"evidence":[],'
            '"rule_basis":["任务书 9(3)；领券和拍链接不是风险"],'
            '"explanation":"有条件表述加普通优惠。"}'
        ),
    ),
    (
        "【口播 ASR】\n今天讲用法。\n【画面 OCR】\n行业天花板 最后几单",
        (
            '{"risk_labels":["存在夸大功效","存在诱导消费"],'
            '"evidence":['
            '{"label":"存在夸大功效","text":"行业天花板","source":"visual"},'
            '{"label":"存在诱导消费","text":"最后几单","source":"visual"}'
            "],"
            '"rule_basis":["风险常在画面叠字"],'
            '"explanation":"口播无风险，画面已有天花板和最后几单。"}'
        ),
    ),
    (
        "【口播 ASR】\n\n【画面 OCR】\n",
        (
            '{"risk_labels":["无法判断"],"evidence":[],'
            '"rule_basis":["任务书 9(5)"],'
            '"explanation":"无可用原文。"}'
        ),
    ),
)

assert RISK_LABELS
assert LABEL_NORMAL and LABEL_UNKNOWN
