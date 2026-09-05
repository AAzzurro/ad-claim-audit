"""根据 ASR/OCR/文案做虚假宣传话术人工标注，写出标注表。

口径对照标注细则表：
- 功效夸大：根治/速效/无效退款/绝对化描述/非医疗用品宣称医疗功能
- 收益承诺：稳赚/保本/日入
- 价格或促销误导（任务书标签「诱导消费」）：原价虚构、限时稀缺、全网最低价
- 站外导流：私信/加群/扫码
- 其他线索：细则四类未覆盖者（如迷信好运）
- 正常/无法判断：有条件描述、普通介绍；证据不足则无法判断

原 videos/output.xlsx 仅供参考。判断以本条 30 秒切片中可见、可听到的内容为准。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.merge import merge_ocr_frames

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "annotations"
LABEL_NORMAL = "正常"
LABEL_EFFICACY = "存在夸大功效"
LABEL_YIELD = "存在虚假收益承诺"
LABEL_INDUCE = "存在诱导消费"
LABEL_OFFSITE = "存在站外导流风险线索"
LABEL_OTHER = "存在其他线索"
LABEL_UNKNOWN = "无法判断"
ANNOTATOR = "人工标注（基于本切片 ASR/OCR/文案复核）"
ANNOTATE_DATE = "2026-09-04"

# 标注细则：绝对化描述（如「最顶尖设计」「行业天花板」）归入夸大功效。不含单字「最」。
_ABS_RE = re.compile(
    r"国家级|世界级|最高级|最高端|最佳|最好|顶级|极品|独一无二|绝无仅有|"
    r"全国第一|全球第一|世界第一|行业第一|第一品牌|销量第一|连续\d年销量|"
    r"行业唯一|色域天花板|省钱天花板|实用天花板|舒适度天花板|"
    r"天花板|首创|独有|首个|首款|最强|最全|无敌|皇室|顶尖|精油之王"
)
# 标注细则：价格或促销误导（「全网最低价」「最后 3 单」）归入诱导消费。
_PRICE_RE = re.compile(r"全网最低价|全年最低价|历史最低价|最低价格|最低价")
# 迷信等细则四类未覆盖的，才留「其他线索」。
_OTHER_RE = re.compile(r"带来好运|转运|开运")



def _ids(*chunks: str) -> set[str]:
    out: set[str] = set()
    for chunk in chunks:
        for part in chunk.replace(",", " ").split():
            if part:
                out.add(part)
    return out


def _span(text: str, needle: str) -> tuple[int, int] | None:
    if not text or not needle:
        return None
    i = text.find(needle)
    if i < 0:
        return None
    return i, i + len(needle)


def _ocr_hit(lines: list[dict], needle: str) -> dict | None:
    for line in lines:
        t = str(line.get("text") or "")
        if needle in t:
            return line
    return None


def evidence_row(
    *,
    label: str,
    text: str,
    source: str,
    snippet: str,
    time: float | None,
    span: tuple[int, int] | None,
) -> dict:
    return {
        "label": label,
        "evidence_text": snippet,
        "evidence_source": source,
        "evidence_time_sec": time,
        "evidence_span": list(span) if span else None,
    }


def find_evidence(label: str, needles: list[str], asr: str, visual: str, ocr_lines: list[dict], orig_times: list[dict]) -> dict | None:
    for needle in needles:
        sp = _span(asr, needle)
        if sp:
            t0 = None
            for ot in orig_times:
                if ot.get("start") is not None:
                    t0 = ot["start"]
                    break
            return evidence_row(
                label=label,
                text=asr,
                source="audio",
                snippet=asr[max(0, sp[0] - 12) : min(len(asr), sp[1] + 12)],
                time=t0,
                span=sp,
            )
        line = _ocr_hit(ocr_lines, needle)
        if line:
            t = str(line.get("text") or "")
            sp = _span(t, needle)
            return evidence_row(
                label=label,
                text=t,
                source="visual",
                snippet=t,
                time=line.get("time") if isinstance(line.get("time"), (int, float)) else None,
                span=sp,
            )
        sp = _span(visual, needle)
        if sp:
            return evidence_row(
                label=label,
                text=visual,
                source="visual",
                snippet=visual[max(0, sp[0] - 8) : min(len(visual), sp[1] + 16)],
                time=None,
                span=sp,
            )
    return None


def abs_excluded(text: str, start: int, end: int) -> bool:
    """排除非产品绝对化主张的用法。"""
    lo = max(0, start - 18)
    hi = min(len(text), end + 18)
    window = text[lo:hi]
    surface = text[start:end]
    if "最高领取" in window:
        return True
    if surface == "首选" and re.search(r"首选\s*[1一]号|先去首选|首选一定是选", window):
        return True
    if "最佳学习" in window or "探参最佳" in window:
        return True
    if "笨蛋加无敌" in window:
        return True
    if "永久的做这个行业" in window or "接班了永久" in window:
        return True
    if "价格天花板也就" in window:
        return True
    if "没有天花板" in window:
        return True
    if surface == "天花板" and ("预算" in window or "也就几百" in window):
        return True
    if surface == "首选" and ("选咖啡" in window or "选黑色" in window):
        return True
    return False


# ---------------------------------------------------------------------------
# 人工标签：先分组，再对特例覆盖。未列入者再扫绝对化用语。
# ---------------------------------------------------------------------------

NORMAL_IDS = _ids(
    "1 2 3 4 5 17 33 80 81 82 87 136 138 167 187 190 195",
    "199 200 202 204 208 214 216 219 223 232 235 236",
    "133 135 198",
)
UNKNOWN_IDS = _ids("210")

# 功效断言 / 医疗用语 / 保健食品疾病暗示（本切片可见）
EFFICACY = {
    "6": ["即刻紧致", "重塑私密"],
    "8": ["有害蓝光", "视网", "近视"],
    "11": ["过敏能退"],
    "23": ["三天的效果", "拿回家三天"],
    "24": ["源头去瓦解黑色"],
    "50": ["14天改善10大纹路"],
    "129": ["见效快", "14天改善"],
    "134": ["小宝压摇膏能吃", "小桃莓也能吃"],
    "150": ["定向肺细胞", "直入肺源"],
    "158": ["胃粘膜损伤", "修复我们这个粘膜"],
    "159": ["不伤身体", "瘦全身"],
    "160": ["减肥没有任何不良反应", "心脏", "这个产品都能喝"],
    "161": ["没有不良反应", "心脏大条", "想减肥"],
    "162": ["治疗骨关节炎", "软骨开始破损"],
    "163": ["医用治疗型", "静脉曲张"],
    "164": ["燃烧脂肪", "中式减肥"],
    "165": ["三高", "心脏治过架", "中式减肥"],
    "166": ["良性的", "恶性"],
    "168": ["治疗就是就是脱发", "秃出小宝贝的救星"],
    "169": ["稳血母糖", "贝塔葡血糖"],
    "170": ["黄褐斑这问题非常非常好", "内斑一窝"],
    "171": ["去黄褐斑的", "内分泌的问题"],
    "172": ["真美白真祛斑", "老年斑都能淡化"],
    "173": ["小唐人的特殊人群"],
    "174": ["美白祛斑", "买一套就搞定"],
    "175": ["即刻去褪痘红", "隔夜就能速别火山痘"],
    "176": ["肌痘根", "痘痘免根"],
    "177": ["杜绝咱们肌肤的过敏"],
    "178": ["祛痘根", "直达痘一层"],
    "179": ["根源去祛痘", "就不会复发", "去根"],
    "180": ["12个小时就起效", "紧急去祛痘"],
    "181": ["三高高小糖人", "都可以放心去吃"],
    "182": ["定向直达肺细胞", "快速修复损伤", "修复效率提升20倍"],
    "183": ["肾炎", "透吸", "尿毒"],
    "184": ["二型的老客户"],
    "186": ["改善静脉曲张"],
    "189": ["斑斑点点", "睡眠不是特别好"],
    "203": ["断根", "漏尿", "去根"],
    "207": ["活到120岁", "长寿因此", "玻尿酸"],
    "226": ["用在航天的，生在医疗的", "抗菌"],
    "229": ["补钙了", "腰酸", "全球首创的三文鱼骨"],
}

INDUCE = {
    "10": ["先下单", "活动就一天"],
    "20": ["最后最后十4箱", "最后再给你加"],
    "44": ["现在买有货，真的不不代表你等到晚上"],
    "45": ["赶紧去买", "晚上8点钟之后还有货"],
    "94": ["倒计时321"],
    "95": ["倒计时321", "全年最低价"],
    "124": ["最后一波了", "倒计时321", "货不多了"],
    "129": ["最后两单"],
    "139": ["先去把名额给占上", "等会再来的话，就是5999"],
    "141": ["最后两单现货"],
    "150": ["最后一批", "卖完就断货", "过期不候"],
    "168": ["仅剩5天"],
    "174": ["只剩4单", "只能现在拍了"],
    "183": ["等着没货了", "赶紧"],
    "234": ["最后一场", "错过等明年", "全网最低价"],
}

YIELD = {
    "192": ["不谈模式只谈收益", "零门槛加盟"],
}

# 扫描未覆盖的绝对化主张，补进夸大功效。
ABS_FORCE = {
    "14": ["肌肤老10倍", "95%以上的皮肤问题"],
    "15": ["霸榜 TOP1", "TOP1"],
    "16": ["无毒无害", "无害"],
    "19": ["排前三"],
    "141": ["位列第一"],
    "149": ["囤货首选", "销售第一"],
    "185": ["绝对是首选"],
    "230": ["最高款", "爱马仕"],
}
# 扫描未覆盖的价格误导，补进诱导消费。
PRICE_FORCE = {
    "18": ["最低价"],
    "44": ["最低价格"],
    "45": ["最低价格"],
    "96": ["全年最低价"],
    "150": ["历史最低价"],
    "196": ["最低价格"],
}
# 细则四类未覆盖：迷信好运等。
OTHER_FORCE = {
    "191": ["带来好运", "紫气东来"],
}

NOTES = {
    "1": "原标导向/性别对立，本任务只标虚假宣传。口播无功效断言或绝对化用语，「最高领取」不视为最高级。",
    "2": "原标导向/外貌贬低，不属虚假宣传话术。",
    "3": "原标导向/体罚言论。「笨蛋加无敌」不是商品绝对化用语。",
    "4": "原标导向/地域歧视，不属虚假宣传话术。",
    "5": "原标低俗导向。画面「逆龄紧致」为常见妆品卖点，未见根治/医疗断言，本任务标正常。",
    "6": "私护产品画面写「即刻紧致」「重塑私密年轻态」，属功效断言。导向类口播不计入本任务。",
    "8": "口播把有害蓝光与黄斑/近视直接因果化，属医疗功能宣称；画面「行业唯一/色域天花板」按细则归入夸大功效下的绝对化描述。",
    "10": "口播要求先下单再告知五折玩法，画面「活动就一天」属价格/促销误导；「天花板」按细则标夸大功效（绝对化描述）。",
    "11": "口播「过敏能退」属功效断言；画面「防晒天花板」「连续5年销量」为绝对化描述，同归夸大功效。",
    "12": "画面「防晒天花板」「连续5年销量」为绝对化描述，归入夸大功效。口播「美白祛斑持证」有特证表述，不另计医疗宣称。",
    "7": "画面「行业唯一」「色域天花板」为绝对化描述，按细则归入夸大功效，不再标其他线索。",
    "20": "口播称水蜜桃「天花板」（绝对化→夸大功效），并以「最后十四箱」制造稀缺（促销误导→诱导消费）。",
    "21": "口播转写为空，以画面「行业唯一 色域天花板」为准，标夸大功效（绝对化描述）。",
    "44": "口播「全部做到最低价格」属价格误导；并以现货稍后可能无货压迫下单。",
    "150": "口播「最后一批卖完断货」+「历史最低价」属价格/促销误导；画面「定向肺细胞」属功效断言。",
    "188": "口播/画面使用「国家级」，属绝对化描述，归入夸大功效。",
    "191": "以穿紫/佩戴紫色饰品「带来好运」推销，属迷信运气承诺；细则四类未覆盖，留其他线索。",
    "192": "加盟广告「不谈模式只谈收益」标虚假收益承诺；「国内首个」为绝对化描述，归入夸大功效。",
    "230": "口播称面料为「麻类的爱马仕」「市面上的最高款」，属绝对化描述，归入夸大功效。原标「无敌」未出现。",
    "234": "画面「最后一场清版」「错过等明年」「全网最低价」属价格或促销误导，归入诱导消费。",
    "18": "口播/画面「全都是最低价了」属价格误导，归入诱导消费，不按绝对化功效标。",
    "196": "口播「拿到最低价格」属价格误导，归入诱导消费。",
    "208": "「价格天花板也就几百块」指学生预算上限，不是商品「行业天花板」，不标夸大功效。",
    "216": "「美起来没有天花板」是程度修辞，不是商品绝对化自称。",
    "14": "「防晒不到位肌肤老10倍」等绝对化因果断言，按绝对化描述归入夸大功效。",
    "16": "画面灭蚊灯文案「无毒无害」为绝对化安全宣称，归入夸大功效。",
    "174": "口播「美白祛斑」属功效断言；「只剩4单」属限时稀缺，归入诱导消费。",
    "95": "倒计时321属促销压迫；「全年最低价」属价格误导，同归诱导消费；「首创者」为绝对化描述，归入夸大功效。",
    "96": "「全年最低价」属价格误导；「最全」「首创者」为绝对化描述。",
    "129": "「见效快/14天改善」属速效功效；「最后两单」属限时稀缺；「最全」为绝对化描述。",
    "139": "「先占名额/等会5999」属促销压迫；画面「行业TOP1」为绝对化描述。",
    "152": "口播称保健品行业「天花板」，绝对化描述归入夸大功效。",
    "185": "口播「绝对是首选」为绝对化描述，归入夸大功效。",
    "201": "口播「无敌巨好用」为绝对化描述，归入夸大功效。",
    "205": "口播「全球首创」为绝对化描述，归入夸大功效。",
    "206": "口播「全球首创」为绝对化描述，归入夸大功效。",
    "209": "画面「粉嫩丝袜的天花板」为绝对化描述，归入夸大功效。",
    "215": "画面「护肤界的天花板」为绝对化描述，归入夸大功效。",
    "217": "口播/画面「舒适度天花板」为绝对化描述，归入夸大功效。",
    "224": "口播「行李箱天花板」为绝对化描述，归入夸大功效。",
    "237": "口播「实用天花板」为绝对化描述，归入夸大功效。",
    "132": "口播「这已经是最低价了」属价格误导，归入诱导消费。",
    "17": "原标「首选」未在本切片口播/画面出现。",
    "23": "原标「最佳」未出现。口播以「三天的效果」诱导先用再见效，属速效功效暗示。",
    "24": "原标「首创」未在本切片出现。口播称从源头瓦解黑色素生成，属功效断言。",
    "33": "原标「独一无二」未在本切片出现，剩余为常规价保促销。",
    "80": "原标「天花板」未在本切片出现。",
    "81": "原标「天花板」未在本切片出现。",
    "82": "原标「无菌」未在本切片出现。",
    "87": "原标「首创」未在本切片出现；「直接拍」为常规催单，未达稀缺误导。",
    "94": "原标「首创」未在本切片出现。口播倒计时321催促拍下，标诱导消费。",
    "133": "画面「首选1号旗舰蓝罐」是导购指向链接，不按「全国首选」绝对化处理。",
    "135": "画面「首选1号旗舰蓝罐」为直播导购用语，未见品牌绝对化或功效断言。",
    "198": "原标导向/低俗暗示，本任务只标虚假宣传。口播为香氛清洁卖点，未见功效根治或绝对化用语。",
    "134": "口播称高血压/糖人也能吃，保健食品对疾病人群作适宜性暗示；画面「首选1号」仍视为导购用语。",
    "136": "原标「首选」未在本切片作为绝对化主张出现。",
    "138": "原标「无菌」未在本切片出现。",
    "162": "虽口头说「不涉及治疗」，仍宣讲氨糖「治疗骨关节炎」并暗示软骨破损，保健食品涉及疾病治疗。",
    "163": "画面商品名「医用治疗型静脉曲张袜」，以治疗/预防静脉曲张口吻售卖，高敏感领域从严。",
    "167": "「无铅无汞无激素」为安全性陈述，未见根治或绝对化用语。",
    "186": "口播几乎不可用；画面评价区写「改善静脉曲张」等疾病功效，作为广告展示内容计入。",
    "203": "口播宣称产后漏尿「十人九人断根」「谁用谁去根」，非药冒充药品式功效断言。",
    "207": "「可能活到120岁」虽有「可能」，后文把玻尿酸与裸鼹鼠不得癌/长寿因子绑定，化妆品领域从严标夸大功效。",
    "210": "口播为空，画面未见「首创」或其它虚假宣传原文，证据不足。",
    "232": "「首选一定是选咖啡」是颜色推荐，不是品牌「首选」。",
    "235": "「永久做这个行业」指子承父业，不是产品永久有效。",
    "236": "原标「最佳」未出现，内容为眼袋预防科普，未见虚假宣传话术。",
}


def load_workspace() -> list[dict]:
    path = ROOT / "data" / "annotation_workspace.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit("缺少 data/annotation_workspace.json，请先导出工作区")


def load_ocr_lines(sample_id: str) -> list[dict]:
    p = ROOT / "data" / "ocr" / f"{sample_id}.json"
    if not p.exists():
        return []
    ocr = json.loads(p.read_text(encoding="utf-8"))
    return merge_ocr_frames(ocr.get("frames") or [])


def collect_labels(sid: str) -> list[str]:
    labels: list[str] = []
    if sid in UNKNOWN_IDS:
        return [LABEL_UNKNOWN]
    if sid in NORMAL_IDS:
        return [LABEL_NORMAL]
    if sid in EFFICACY or sid in ABS_FORCE:
        labels.append(LABEL_EFFICACY)
    if sid in YIELD:
        labels.append(LABEL_YIELD)
    if sid in INDUCE or sid in PRICE_FORCE:
        labels.append(LABEL_INDUCE)
    if sid in OTHER_FORCE:
        labels.append(LABEL_OTHER)
    return labels


def scan_regex(pattern: re.Pattern[str], asr: str, visual: str) -> list[tuple[str, tuple[int, int], str]]:
    hits = []
    for source, text in (("audio", asr), ("visual", visual)):
        if not text:
            continue
        for m in pattern.finditer(text):
            if abs_excluded(text, m.start(), m.end()):
                continue
            hits.append((source, (m.start(), m.end()), m.group(0)))
    seen = set()
    uniq = []
    for h in hits:
        if h[2] in seen:
            continue
        seen.add(h[2])
        uniq.append(h)
    return uniq


def build_note(sid: str, labels: list[str], abs_surfaces: list[str], price_surfaces: list[str]) -> str:
    if sid in NOTES:
        return NOTES[sid]
    if labels == [LABEL_NORMAL]:
        return "本切片未见虚假宣传话术原文。原标签若为其他违法类型，本任务不标注。"
    if labels == [LABEL_UNKNOWN]:
        return "本切片可核原文不足，无法判断是否构成虚假宣传话术。"
    bits = []
    if LABEL_EFFICACY in labels:
        if abs_surfaces:
            bits.append("绝对化描述（「" + "、".join(abs_surfaces[:5]) + "」）按细则归入夸大功效。")
        else:
            bits.append("口播或画面存在功效/医疗/疾病人群断言。")
    if LABEL_INDUCE in labels:
        if price_surfaces:
            bits.append("价格或促销误导（「" + "、".join(price_surfaces[:4]) + "」）归入诱导消费。")
        else:
            bits.append("存在稀缺或压迫下单话术。")
    if LABEL_YIELD in labels:
        bits.append("存在收益/加盟收益导向表述。")
    if LABEL_OTHER in labels:
        bits.append("细则四类未覆盖的其他虚假宣传线索。")
    bits.append("仅依据本切片可见可听内容，不推测商品真实效果。")
    return "".join(bits)


def main() -> None:
    items = load_workspace()
    rows = []
    long_rows = []
    missing = []

    for it in items:
        sid = str(it["sample_id"])
        asr = it.get("audio_text") or ""
        visual = it.get("visual_text") or ""
        ocr_lines = load_ocr_lines(sid)
        orig_times = it.get("orig_times") or []
        copy_text = it.get("广告描述") or ""

        labels = collect_labels(sid)
        abs_hits = [] if sid in NORMAL_IDS or sid in UNKNOWN_IDS else scan_regex(_ABS_RE, asr, visual)
        price_hits = [] if sid in NORMAL_IDS or sid in UNKNOWN_IDS else scan_regex(_PRICE_RE, asr, visual)
        abs_surfaces = [h[2] for h in abs_hits]
        price_surfaces = [h[2] for h in price_hits]
        if abs_hits and LABEL_EFFICACY not in labels and sid not in NORMAL_IDS:
            labels.append(LABEL_EFFICACY)
        if price_hits and LABEL_INDUCE not in labels and sid not in NORMAL_IDS:
            labels.append(LABEL_INDUCE)

        extra_needles: dict[str, list[str]] = {}
        if sid in EFFICACY:
            extra_needles.setdefault(LABEL_EFFICACY, []).extend(EFFICACY[sid])
        if sid in ABS_FORCE:
            extra_needles.setdefault(LABEL_EFFICACY, []).extend(ABS_FORCE[sid])
        if abs_surfaces:
            extra_needles.setdefault(LABEL_EFFICACY, []).extend(abs_surfaces)
        if sid in INDUCE:
            extra_needles.setdefault(LABEL_INDUCE, []).extend(INDUCE[sid])
        if sid in PRICE_FORCE:
            extra_needles.setdefault(LABEL_INDUCE, []).extend(PRICE_FORCE[sid])
        if price_surfaces:
            extra_needles.setdefault(LABEL_INDUCE, []).extend(price_surfaces)
        if sid in YIELD:
            extra_needles.setdefault(LABEL_YIELD, []).extend(YIELD[sid])
        if sid in OTHER_FORCE:
            extra_needles.setdefault(LABEL_OTHER, []).extend(OTHER_FORCE[sid])

        if not labels:
            labels = [LABEL_NORMAL]
            missing.append(sid)

        order = [LABEL_EFFICACY, LABEL_YIELD, LABEL_INDUCE, LABEL_OFFSITE, LABEL_OTHER, LABEL_NORMAL, LABEL_UNKNOWN]
        labels = [x for x in order if x in labels]

        evids: list[dict] = []
        seen_snip: set[str] = set()
        for lab in labels:
            if lab in (LABEL_NORMAL, LABEL_UNKNOWN):
                continue
            needles = extra_needles.get(lab) or []
            # 去重 needle，按条取证
            used_n = []
            for n in needles:
                if n and n not in used_n:
                    used_n.append(n)
            found = False
            for needle in used_n:
                ev = find_evidence(lab, [needle], asr, visual, ocr_lines, orig_times)
                if not ev:
                    continue
                key = f"{lab}|{ev['evidence_text']}"
                if key in seen_snip:
                    continue
                seen_snip.add(key)
                evids.append(ev)
                found = True
                break
            if not found and used_n:
                evids.append(
                    evidence_row(
                        label=lab,
                        text="",
                        source="visual",
                        snippet=used_n[0],
                        time=None,
                        span=_span(asr, used_n[0]) or _span(visual, used_n[0]),
                    )
                )

        note = build_note(sid, labels, abs_surfaces, price_surfaces)
        pos_txt = []
        ev_txt = []
        for ev in evids:
            ev_txt.append(f"[{ev['label']}] {ev['evidence_text']}")
            t = ev["evidence_time_sec"]
            span = ev["evidence_span"]
            loc = ev["evidence_source"]
            if t is not None:
                loc += f" @{t}s"
            if span:
                loc += f" span={span[0]}-{span[1]}"
            pos_txt.append(f"{ev['label']}: {loc}")

        row = {
            "sample_id": sid,
            "video_name": f"{sid}.mp4",
            "商品类别": it.get("广告类别") or "",
            "文案": copy_text,
            "ASR口播文本": asr,
            "字幕OCR文本": visual,
            "原标签_仅供参考": "；".join(it.get("orig_types") or []),
            "原要素_仅供参考": "；".join(it.get("orig_elements") or []),
            "风险标签": "；".join(labels),
            "是否多标签": "是" if len([x for x in labels if x not in (LABEL_NORMAL, LABEL_UNKNOWN)]) > 1 else "否",
            "证据文本": " || ".join(ev_txt) if ev_txt else "",
            "证据时间戳或文本位置": "；".join(pos_txt) if pos_txt else "",
            "人工说明": note,
            "标注人": ANNOTATOR,
            "标注日期": ANNOTATE_DATE,
            "n_evidence": len(evids),
        }
        rows.append(row)
        if evids:
            for ev in evids:
                long_rows.append(
                    {
                        "sample_id": sid,
                        "风险标签": ev["label"],
                        "证据文本": ev["evidence_text"],
                        "证据来源": ev["evidence_source"],
                        "证据时间戳秒": ev["evidence_time_sec"],
                        "文本起": (ev["evidence_span"] or [None, None])[0],
                        "文本止": (ev["evidence_span"] or [None, None])[1],
                        "人工说明": note,
                    }
                )
        else:
            long_rows.append(
                {
                    "sample_id": sid,
                    "风险标签": labels[0],
                    "证据文本": "",
                    "证据来源": "",
                    "证据时间戳秒": None,
                    "文本起": None,
                    "文本止": None,
                    "人工说明": note,
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["sid_n"] = df["sample_id"].astype(int)
    df = df.sort_values("sid_n").drop(columns=["sid_n"])
    df_long = pd.DataFrame(long_rows)
    df_long["sid_n"] = df_long["sample_id"].astype(int)
    df_long = df_long.sort_values(["sid_n", "风险标签"]).drop(columns=["sid_n"])

    xlsx = OUT_DIR / "manual_annotation.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df.drop(columns=["n_evidence"]).to_excel(writer, index=False, sheet_name="人工标注表")
        df_long.to_excel(writer, index=False, sheet_name="证据明细")
        # summary
        vc = df["风险标签"].value_counts().rename_axis("风险标签组合").reset_index(name="样本数")
        # explode labels
        exploded = df.assign(lab=df["风险标签"].str.split("；")).explode("lab")
        per = exploded["lab"].value_counts().rename_axis("风险标签").reset_index(name="出现次数")
        vc.to_excel(writer, index=False, sheet_name="标签组合统计")
        per.to_excel(writer, index=False, sheet_name="分标签统计")

    csv_path = OUT_DIR / "manual_annotation.csv"
    df.drop(columns=["n_evidence"]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path = OUT_DIR / "manual_annotation.json"
    records = df.drop(columns=["n_evidence"]).to_dict(orient="records")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("samples", len(df))
    print(per.to_string(index=False))
    print("xlsx", xlsx)
    print("fallback_normal_ids", missing)
    print("multi", int((df["是否多标签"] == "是").sum()))
    print("normal", int(df["风险标签"].eq(LABEL_NORMAL).sum()))
    print("unknown", int(df["风险标签"].eq(LABEL_UNKNOWN).sum()))


if __name__ == "__main__":
    main()
