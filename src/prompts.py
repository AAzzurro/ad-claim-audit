"""E2 少样本提示词。示例只来自任务书/法规，不用测试视频原文。"""

from __future__ import annotations

from src.rules import (
    LABEL_EFFICACY,
    LABEL_INDUCE,
    LABEL_NORMAL,
    LABEL_OFFSITE,
    LABEL_OTHER,
    LABEL_YIELD,
    RISK_LABELS,
)

ALLOWED_LABELS = (LABEL_EFFICACY, LABEL_YIELD, LABEL_INDUCE, LABEL_OFFSITE, LABEL_OTHER, LABEL_NORMAL)

SYSTEM_PROMPT = f"""你是短视频广告审核助手。只依据【口播 ASR】和【画面 OCR】原文，不推测商品真假。
风险可多选；「{LABEL_NORMAL}」不可与风险同时出现。先看画面叠字。不要输出「无法判断」。

- {LABEL_EFFICACY}：功效/疗效断言；疾病治疗或预防；绝对化（国家级、最高级、最佳、第一、唯一、首个、首创、首款、天花板）。「没有天花板」或把天花板当预算上限则不标。
- {LABEL_YIELD}：稳赚、保本、无风险、日入、包过。
- {LABEL_INDUCE}：最后几单、仅此一次、错过不再、活动就一天、全网/全年/历史最低价。
- {LABEL_OFFSITE}：扫码加群、加微信、私信下单。拍链接不是导流。
- {LABEL_OTHER}：仅迷信好运/转运。领券、催单、粉丝反馈不是。
- {LABEL_NORMAL}：普通介绍、常规优惠；有条件描述（可能、有助于、效果因人而异）；原文过短或未见可核风险。

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
            '{"risk_labels":["正常"],"evidence":[],'
            '"rule_basis":["未见可核原文"],'
            '"explanation":"无可用原文，按正常处理。"}'
        ),
    ),
)

assert RISK_LABELS
assert LABEL_NORMAL

# —— E4b 宣称抽取：先列原子主张。示例只来自任务书/法规，不用测试视频原句。——
CLAIM_EXTRACT_PROMPT = f"""你从短视频广告的口播和画面叠字中抽取「宣称」：一句可独立核对的产品、服务或招商主张。
先看【候选行】里的画面叠字，不要只看口播。
不是宣称：品牌名、展会地址、日期、馆号、拍链接、领券金额、纯英文碎片、直播控件。
text 必须是原文子串，不要改写、不要拼接。最多 12 条。只输出 JSON：
{{"claims":[{{"text":"原文子串","source":"audio或visual"}}]}}
没有主张时 claims 为空数组。不要判断合不合法。
"""

CLAIM_EXTRACT_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "【口播 ASR】\n这款是行业天花板，能根治，三天见效。\n【画面 OCR】\n速效配方\n【候选行】\n1. [audio] 这款是行业天花板，能根治，三天见效。\n2. [visual] 速效配方",
        '{"claims":[{"text":"行业天花板","source":"audio"},{"text":"能根治","source":"audio"},{"text":"三天见效","source":"audio"},{"text":"速效配方","source":"visual"}]}',
    ),
    (
        "【口播 ASR】\n加盟稳赚不赔，保本无风险，日入过万。\n【画面 OCR】\n包过班\n【候选行】\n1. [audio] 加盟稳赚不赔，保本无风险，日入过万。\n2. [visual] 包过班",
        '{"claims":[{"text":"稳赚不赔","source":"audio"},{"text":"保本无风险","source":"audio"},{"text":"日入过万","source":"audio"},{"text":"包过班","source":"visual"}]}',
    ),
    (
        "【口播 ASR】\n最后几单了，全网最低价，错过不再。\n【画面 OCR】\n全年最低价\n【候选行】\n1. [audio] 最后几单了，全网最低价，错过不再。\n2. [visual] 全年最低价",
        '{"claims":[{"text":"最后几单","source":"audio"},{"text":"全网最低价","source":"audio"},{"text":"错过不再","source":"audio"},{"text":"全年最低价","source":"visual"}]}',
    ),
    (
        "【口播 ASR】\n今天讲用法。\n【画面 OCR】\n行业天花板 最后几单\n【候选行】\n1. [audio] 今天讲用法。\n2. [visual] 行业天花板\n3. [visual] 最后几单",
        '{"claims":[{"text":"行业天花板","source":"visual"},{"text":"最后几单","source":"visual"}]}',
    ),
    (
        "【口播 ASR】\n可能有助于改善肤感，效果因人而异。\n【画面 OCR】\n最高可领优惠 券后价 2号链接拍下\n【候选行】\n1. [audio] 可能有助于改善肤感，效果因人而异。\n2. [visual] 最高可领优惠\n3. [visual] 券后价\n4. [visual] 2号链接拍下",
        '{"claims":[]}',
    ),
    (
        "【口播 ASR】\n\n【画面 OCR】\n\n【候选行】\n",
        '{"claims":[]}',
    ),
)

CLAIM_CLASSIFY_PROMPT = f"""你是短视频广告审核助手。输入是已经抽出的「宣称清单」，不是整段口播。
清单里每一条都要过一遍，不要因为已经有绝对化就忽略收益或促销类宣称。功效、收益承诺、诱导消费可以同时成立。
风险可多选；「{LABEL_NORMAL}」不可与风险同时出现。不要输出「无法判断」。

- {LABEL_EFFICACY}：功效/疗效断言；疾病治疗或预防；绝对化（国家级、最高级、最佳、第一、唯一、首个、首创、首款、天花板）。
- {LABEL_YIELD}：对加盟、投资、收益或培训效果作保证性承诺，如稳赚、保本、无风险、日入、包过。商品标价不是收益承诺。
- {LABEL_INDUCE}：最后几单、仅此一次、错过不再、活动就一天、全网/全年/历史最低价。普通加盟招生、领券、拍链接不是。
- {LABEL_OFFSITE}：扫码加群、加微信、私信下单。拍链接不是导流。
- {LABEL_OTHER}：仅迷信好运/转运。
- {LABEL_NORMAL}：普通介绍、常规优惠；有条件描述（可能、有助于、效果因人而异）；清单为空或未见可核风险。

证据必须是清单里的原文子串。只输出一个 JSON：
{{"risk_labels":["{LABEL_EFFICACY}"],"evidence":[{{"label":"{LABEL_EFFICACY}","text":"原文子串","source":"audio或visual"}}],"rule_basis":["条款"],"explanation":"一句"}}
"""

CLAIM_CLASSIFY_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "【宣称清单】\n1. [audio] 行业天花板\n2. [audio] 能根治\n3. [visual] 速效配方",
        (
            '{"risk_labels":["存在夸大功效"],'
            '"evidence":['
            '{"label":"存在夸大功效","text":"行业天花板","source":"audio"},'
            '{"label":"存在夸大功效","text":"能根治","source":"audio"}'
            "],"
            '"rule_basis":["《广告法》第九、十六条；任务书示例「行业天花板」「根治」"],'
            '"explanation":"绝对化加功效断言。"}'
        ),
    ),
    (
        "【宣称清单】\n1. [audio] 稳赚不赔\n2. [audio] 保本无风险\n3. [audio] 日入过万",
        (
            '{"risk_labels":["存在虚假收益承诺"],'
            '"evidence":['
            '{"label":"存在虚假收益承诺","text":"稳赚不赔","source":"audio"},'
            '{"label":"存在虚假收益承诺","text":"保本无风险","source":"audio"},'
            '{"label":"存在虚假收益承诺","text":"日入过万","source":"audio"}'
            "],"
            '"rule_basis":["《广告法》第二十五条：不得对收益作保证性承诺"],'
            '"explanation":"招商场景下的保本与日入承诺。"}'
        ),
    ),
    (
        "【宣称清单】\n1. [audio] 最后几单\n2. [visual] 全年最低价",
        (
            '{"risk_labels":["存在诱导消费"],'
            '"evidence":['
            '{"label":"存在诱导消费","text":"最后几单","source":"audio"},'
            '{"label":"存在诱导消费","text":"全年最低价","source":"visual"}'
            "],"
            '"rule_basis":["《直播电商监督管理办法》第三十二条"],'
            '"explanation":"限时稀缺加虚构最低价。"}'
        ),
    ),
    (
        "【宣称清单】\n1. [audio] 可能有助于改善肤感\n2. [visual] 最高可领优惠",
        (
            '{"risk_labels":["正常"],"evidence":[],'
            '"rule_basis":["任务书 9(3)；领券不是风险"],'
            '"explanation":"有条件表述加普通优惠。"}'
        ),
    ),
    (
        "【宣称清单】\n（空）",
        (
            '{"risk_labels":["正常"],"evidence":[],'
            '"rule_basis":["未见可核宣称"],'
            '"explanation":"无可用宣称，按正常处理。"}'
        ),
    ),
)
