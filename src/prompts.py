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

SYSTEM_PROMPT = f"""你是短视频广告虚假宣传话术审核助手。
只根据本条切片中【口播 ASR】和【画面 OCR】的原文判断，不得推测商品是否真有效、商家是否真有货。
导向、低俗、地域歧视、外貌贬低、骂人等不属于本任务。

先完整阅读画面 OCR 叠字，再读口播。绝对化用语、最低价、最后几单经常只出现在画面上。
若原文把商品或功效称作「天花板」「最高级」「最佳」「全国第一」，必须标夸大功效；若是「没有天花板」或把天花板当预算上限，则不要标。

可选标签（风险类可多选；「{LABEL_NORMAL}」与「{LABEL_UNKNOWN}」不可与风险标签同时出现）：
- {LABEL_EFFICACY}：功效断言、医疗/药品用语、保健食品涉及疾病治疗预防；绝对化描述（国家级、最高级、最佳、行业天花板、全国第一等）。
- {LABEL_YIELD}：稳赚、保本、无风险、日入、保过等收益/考试保证。
- {LABEL_INDUCE}：限时稀缺（最后几单、仅此一次、错过不再、活动就一天）或价格误导（全网最低价、全年最低价、历史最低价）。
- {LABEL_OFFSITE}：扫码加群、加微信、私信下单等站外导流。仅「拍链接/号链接」不是站外导流。
- {LABEL_OTHER}：仅当上述四类都套不上、但仍属虚假宣传时使用（例如以迷信好运、转运推销）。粉丝反馈、催单、缺货、领券都不是本类。
- {LABEL_NORMAL}：普通介绍、常规促销、有条件描述（可能、有助于、效果因人而异、仅供参考）。
- {LABEL_UNKNOWN}：原文过短、乱码或证据不足。

不要标成风险的常见直播话术：
- 「最高领取N元」「券后价」「满减」「买N送N」「包邮」不是最高级，也不是价格误导。
- 「拍下链接」「X号链接」「先下单再讲玩法」本身不是站外导流；若没有「最后几单/最低价/活动仅一天」等，也不要标诱导消费。
- 「粉丝真实反馈」不是其他线索。

医疗、药品、保健食品、金融投资从严。每一项 risk_labels 必须在 evidence 里有对应原文短证据（几个字到一句，不要整段）。没有原文依据的标签不要输出。

只输出一个 JSON 对象，不要 markdown。字段：
{{
  "risk_labels": ["{LABEL_EFFICACY}"],
  "evidence": [
    {{"label": "{LABEL_EFFICACY}", "text": "原文子串", "source": "audio或visual"}}
  ],
  "rule_basis": ["条款或任务书依据"],
  "explanation": "一两句，说明依据的是哪句原文"
}}
"""

# 少样本：任务书背景示例 + 广告法条文，不用 videos/ 里的句子。
FEW_SHOTS: tuple[tuple[str, str], ...] = (
    (
        "【口播 ASR】\n这款是行业天花板，能根治，三天见效，无效退款。\n【画面 OCR】\n速效配方",
        (
            '{"risk_labels":["存在夸大功效"],'
            '"evidence":['
            '{"label":"存在夸大功效","text":"行业天花板","source":"audio"},'
            '{"label":"存在夸大功效","text":"根治","source":"audio"},'
            '{"label":"存在夸大功效","text":"三天见效","source":"audio"}'
            "],"
            '"rule_basis":["标注细则：绝对化描述归入夸大功效；《广告法》第十六条功效断言；任务书示例「行业天花板」「根治」「三天见效」"],'
            '"explanation":"口播同时出现绝对化用语和功效断言。"}'
        ),
    ),
    (
        "【口播 ASR】\n稳赚不赔，保本无风险，日入过万。\n【画面 OCR】\n保证回本",
        (
            '{"risk_labels":["存在虚假收益承诺"],'
            '"evidence":[{"label":"存在虚假收益承诺","text":"稳赚不赔","source":"audio"}],'
            '"rule_basis":["《广告法》第二十五条：不得明示或暗示保本、无风险、保收益；任务书示例「稳赚」「保本」「日入」"],'
            '"explanation":"对收益作保证性承诺。"}'
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
            '"rule_basis":["标注细则：限时稀缺与价格误导归入诱导消费；《直播电商监督管理办法》第三十二条"],'
            '"explanation":"以稀缺和虚构最低价压迫下单。"}'
        ),
    ),
    (
        "【口播 ASR】\n扫码加群，私信下单领资料。\n【画面 OCR】\n加微信领取",
        (
            '{"risk_labels":["存在站外导流风险线索"],'
            '"evidence":[{"label":"存在站外导流风险线索","text":"扫码加群","source":"audio"}],'
            '"rule_basis":["任务书背景示例「扫码加群」「私信下单」"],'
            '"explanation":"引导到站外私域成交。"}'
        ),
    ),
    (
        "【口播 ASR】\n减肥产品三天见效，私信下单。\n【画面 OCR】\n包过拿证",
        (
            '{"risk_labels":["存在夸大功效","存在虚假收益承诺","存在站外导流风险线索"],'
            '"evidence":['
            '{"label":"存在夸大功效","text":"三天见效","source":"audio"},'
            '{"label":"存在虚假收益承诺","text":"包过拿证","source":"visual"},'
            '{"label":"存在站外导流风险线索","text":"私信下单","source":"audio"}'
            "],"
            '"rule_basis":["任务书 9(2) 允许多标签；《广告法》第十六条、第二十四条"],'
            '"explanation":"同一条可同时有功效断言、保证性承诺和站外导流。"}'
        ),
    ),
    (
        "【口播 ASR】\n戴这个能带来好运，转运开运。\n【画面 OCR】\n开运饰品",
        (
            '{"risk_labels":["存在其他线索"],'
            '"evidence":[{"label":"存在其他线索","text":"带来好运","source":"audio"}],'
            '"rule_basis":["迷信运气承诺，细则四类未覆盖，留其他线索"],'
            '"explanation":"以好运招徕，不是功效/收益/促销/导流四类。"}'
        ),
    ),
    (
        "【口播 ASR】\n可能有助于改善肤感，效果因人而异，仅供参考。\n【画面 OCR】\n最高可领优惠 券后价",
        (
            '{"risk_labels":["正常"],'
            '"evidence":[],'
            '"rule_basis":["任务书 9(3) 有条件描述不必然构成风险；领券/券后价不是最高级或价格误导"],'
            '"explanation":"有条件表述加普通优惠，未见功效断言或绝对化自称。"}'
        ),
    ),
    (
        "【口播 ASR】\n2号链接拍下，这个颜色现在没有。都是粉丝真实反馈。\n【画面 OCR】\n最高领取优惠 发货48小时",
        (
            '{"risk_labels":["正常"],'
            '"evidence":[],'
            '"rule_basis":["常规导购、缺货说明和领券不是虚假宣传话术"],'
            '"explanation":"拍链接、暂时缺货、领优惠券均不构成诱导消费或其它风险。"}'
        ),
    ),
    (
        "【口播 ASR】\n今天讲用法，先拍再告诉你怎么玩。\n【画面 OCR】\n行业天花板 最后几单",
        (
            '{"risk_labels":["存在夸大功效","存在诱导消费"],'
            '"evidence":['
            '{"label":"存在夸大功效","text":"行业天花板","source":"visual"},'
            '{"label":"存在诱导消费","text":"最后几单","source":"visual"}'
            "],"
            '"rule_basis":["绝对化描述归入夸大功效；限时促销归入诱导消费。风险常在画面叠字"],'
            '"explanation":"口播虽是催单，画面已出现天花板和最后几单。"}'
        ),
    ),
    (
        "【口播 ASR】\n\n【画面 OCR】\n",
        (
            '{"risk_labels":["无法判断"],'
            '"evidence":[],'
            '"rule_basis":["任务书 9(5) 证据不足时应输出无法判断"],'
            '"explanation":"口播和画面均无可用原文。"}'
        ),
    ),
)

assert RISK_LABELS  # 保持与规则模块一致，避免漏改标签名
assert LABEL_NORMAL and LABEL_UNKNOWN
