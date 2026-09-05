"""E1 关键词/规则基线：词表只来自任务书与法规，不从测试视频增补。

来源：
- Knowledge/短视频广告虚假宣传话术检测.pdf（标签、示例话术、有条件描述）
- Knowledge/8中华人民共和国广告法.pdf（第九、十六–十八、二十四、二十五条等）
- Knowledge/直播电商监督管理办法.pdf（第三十二条促销标示、第三十四条虚假宣传）

禁止：根据 videos/、data/merged、data/asr、data/ocr 里出现的句子往词表加词。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 任务书第三节：六类标签 + 「无法判断」
LABEL_NORMAL = "正常"
LABEL_EFFICACY = "存在夸大功效"
LABEL_YIELD = "存在虚假收益承诺"
LABEL_INDUCE = "存在诱导消费"
LABEL_OFFSITE = "存在站外导流风险线索"
LABEL_OTHER = "存在其他线索"
LABEL_UNKNOWN = "无法判断"

RISK_LABELS = (LABEL_EFFICACY, LABEL_YIELD, LABEL_INDUCE, LABEL_OFFSITE, LABEL_OTHER)

# 任务书 9(3)：有条件描述不必然构成风险
_HEDGE_RE = re.compile(r"可能|有助于|部分用户反馈|效果因人而异|仅供参考")


@dataclass(frozen=True)
class Rule:
    pattern: str
    label: str
    rule_basis: str
    compiled: re.Pattern[str]


def _rule(pattern: str, label: str, rule_basis: str) -> Rule:
    return Rule(pattern=pattern, label=label, rule_basis=rule_basis, compiled=re.compile(pattern))


# 每条规则的 rule_basis 写清条款/任务书位置，便于审核复核。
# 标签对照细则：绝对化描述归入夸大功效；「全网最低价」等价格误导归入诱导消费。
RULES: tuple[Rule, ...] = (
    # —— 绝对化描述 → 夸大功效（细则「最顶尖设计」；任务书示例「行业天花板」）——
    # 广告法第九条第三项原文：国家级、最高级、最佳。不用单字「最」（避免「最高领取」）。
    _rule(
        r"国家级|世界级|最高级|最佳|最好|顶级|极品|独一无二|绝无仅有",
        LABEL_EFFICACY,
        "标注细则「功效夸大-绝对化描述」；《广告法》第九条第三项：不得使用「国家级」「最高级」「最佳」等用语",
    ),
    _rule(
        r"全国第一|全球第一|世界第一|行业第一|第一品牌|销量第一",
        LABEL_EFFICACY,
        "标注细则「功效夸大-绝对化描述」；《广告法》第九条第三项「等」：与「最高级/最佳」同类的第一、唯一表述",
    ),
    _rule(
        r"行业天花板|天花板",
        LABEL_EFFICACY,
        "标注细则「功效夸大-绝对化描述」（例「最顶尖设计」）；任务书背景示例「行业天花板」",
    ),
    # —— 夸大功效 / 医疗用语 ——
    _rule(
        r"根治|治愈率|有效率|治愈|痊愈|根除|药到病除|疗效保证|无效退款",
        LABEL_EFFICACY,
        "《广告法》第十六条、第十七条：不得作功效断言、说明治愈率有效率、非药品不得涉及疾病治疗；任务书示例「根治」「无效退款」",
    ),
    _rule(
        r"速效|三天见效|当天见效|立即见效|马上见效|永不反弹|永久有效",
        LABEL_EFFICACY,
        "《广告法》第十六条第一项功效断言；任务书示例「速效」「三天见效」",
    ),
    _rule(
        r"治疗功能|治疗效果|医用|药用|替代药物|代替药物|抗癌|消炎药",
        LABEL_EFFICACY,
        "《广告法》第十七条：禁止其他广告涉及疾病治疗功能或使用医疗用语；第十八条保健食品不得涉及疾病预防治疗",
    ),
    # —— 收益承诺 ——
    _rule(
        r"稳赚不赔|稳赚|保本|保收益|无风险|躺赚|日入|月入万元|保证回本",
        LABEL_YIELD,
        "《广告法》第二十五条：招商等广告不得对收益作保证性承诺，不得明示或暗示保本、无风险、保收益；任务书示例「稳赚」「保本」「日入」",
    ),
    _rule(
        r"包过|保过|不过退款|保证升学|保证拿证",
        LABEL_YIELD,
        "《广告法》第二十四条：教育培训广告不得对升学、考试、证书效果作保证性承诺",
    ),
    # —— 诱导消费：限时稀缺 + 价格误导 ——
    _rule(
        r"最后几单|最后一批|最后一天|仅此一次|错过不再|不买就没",
        LABEL_INDUCE,
        "标注细则「价格或促销误导-限时稀缺」；任务书示例「最后几单」；《直播电商监督管理办法》第三十二条",
    ),
    _rule(
        r"全网最低价|全年最低价|历史最低价|最低价格|最低价",
        LABEL_INDUCE,
        "标注细则「价格或促销误导」示例「全网最低价」；比较价格须有真实基准，不得虚构最低价",
    ),
    # —— 站外导流 ——
    _rule(
        r"扫码加群|加微信|加薇信|加微信群|加v\b|加V\b|私信下单|私信领|复制淘口令|站外联系",
        LABEL_OFFSITE,
        "任务书背景示例「扫码加群」、标注示例「私信下单」",
    ),
)


def hedge_nearby(text: str, start: int, end: int, window: int = 16) -> bool:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    return bool(_HEDGE_RE.search(text[lo:hi]))


def iter_matches(text: str):
    """在一段文本上打规则。同一跨度只保留更长的命中。"""
    if not text:
        return
    found: list[tuple[int, int, Rule, str]] = []
    for rule in RULES:
        for match in rule.compiled.finditer(text):
            found.append((match.start(), match.end(), rule, match.group(0)))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    used: list[tuple[int, int]] = []
    for start, end, rule, surface in found:
        if any(start < u_end and end > u_start for u_start, u_end in used):
            continue
        used.append((start, end))
        yield start, end, rule, surface
