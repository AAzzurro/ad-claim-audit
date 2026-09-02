from __future__ import annotations

import re
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\[\]【】()（）<>《》""\"'`，。！？、…·^~]+")

# 直播控件 / 互动提示。价格、功效、促销留给下游关键词基线。
_UI_RE = re.compile(
    r"关注|点赞|分享|评论|粉丝团|进入直播间|送礼|红包|购物车|在线人数|人观看|开播|"
    r"直播|抖音|快手|小红书|用户等级|榜单|来了|"
    r"人气榜|小时榜|热卖|榜第|"
    r"说点什么|商品信息|"
    r"抽奖|抢券|讲解中|来自推广|来自推|"
    r"库存仅剩|浏览|回头客|买家秀|加入|"
    r"去领取|点右上角|发货须知|换手机拍|领不到券|憨憨点亮了|"
    r"限时挑战|抢现金|卖货频道"
)
_CHROME_EXACT = {
    "维度",
    "内容",
    "默认",
    "主播",
    "西2区",
    "满意",
    "付款!",
    "付款！",
    "抢>",
    "草！",
    "慕!",
    "加?",
    "加？",
}
_ONLY_NOISE_RE = re.compile(r"^[\W_\d]+$|^[0-9.]+[万千万]?$", re.UNICODE)
_ORDER_TIP_RE = re.compile(
    r"(?:\*{2,}.*)?(?:下单|卜单).{0,8}号商品|\*{2,}.*下单|号商品$|"
    r"先下单吗|开了吗|怎么参|有没有活动|中一次"
)
_LIVE_COMMENT_RE = re.compile(r"^(?P<left>.{1,24})[：:]")
# 冒号左侧是短价签/规格，不是「昵称+旗舰店」。
_LABEL_LEFT_RE = re.compile(r"价|券|折|发货|规格|产地|品牌|链接|优惠|活动")
# 商品卡 / 卖点。不要写具体品名（气垫、山茶），否则无法泛化。
_OFFER_RE = re.compile(
    r"价|折|券|¥|元|买.{0,8}送|拍\d|链接|包邮|低价|热销|专场|优惠|"
    r"正品|升级|旗舰|系列|精华|之王|天花板|全网|同款|退货|新品"
)
_GIFT_EXACT_RE = re.compile(r"^礼品\d*$")
_REVIEW_DATE_RE = re.compile(r"^\d+\s*(个月前|天前|小时前)$")


def normalize_key(text: str) -> str:
    text = _PUNCT_RE.sub("", text)
    text = _SPACE_RE.sub("", text)
    return text.strip().lower()


def _box_center(box: Any) -> tuple[float, float] | None:
    if not box or not isinstance(box, (list, tuple)):
        return None
    if not isinstance(box[0], (list, tuple)):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for pt in box:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _canvas_size(frames: list[dict[str, Any]]) -> tuple[float, float]:
    max_x = max_y = 1.0
    for frame in frames:
        for item in frame.get("texts") or []:
            xy = _box_center(item.get("box"))
            if xy:
                max_x = max(max_x, xy[0])
                max_y = max(max_y, xy[1])
    return max_x, max_y


def is_live_comment(text: str) -> bool:
    match = _LIVE_COMMENT_RE.match(text)
    if not match:
        return False
    left = match.group("left")
    if len(left) <= 8 and _LABEL_LEFT_RE.search(left):
        return False
    return True


def _in_comment_or_gift_band(nx: float, ny: float) -> bool:
    """公屏左下、抽奖奖品左上、赠品条底部中间。右侧商品卡不裁。"""
    if nx < 0.50 and ny > 0.70:
        return True
    if nx < 0.22 and 0.20 < ny < 0.50:
        return True
    if 0.48 <= nx <= 0.76 and ny > 0.84:
        return True
    return False


def is_ocr_noise(
    text: str,
    *,
    box: Any = None,
    canvas: tuple[float, float] | None = None,
) -> bool:
    """控件、下单提示、进场、公屏。左下去公屏时放行商品卡/卖点。"""
    if text in _CHROME_EXACT or _GIFT_EXACT_RE.fullmatch(text):
        return True
    if _UI_RE.search(text) or _ONLY_NOISE_RE.fullmatch(text) or _ORDER_TIP_RE.search(text):
        return True
    if _REVIEW_DATE_RE.fullmatch(text):
        return True
    if is_live_comment(text):
        return True
    if canvas and box is not None and not _OFFER_RE.search(text):
        xy = _box_center(box)
        if xy:
            nx, ny = xy[0] / canvas[0], xy[1] / canvas[1]
            if _in_comment_or_gift_band(nx, ny):
                return True
    return False


def _is_near_duplicate(key: str, seen: set[str]) -> bool:
    if key in seen:
        return True
    for prev in seen:
        shorter, longer = (key, prev) if len(key) <= len(prev) else (prev, key)
        if len(shorter) >= 2 and len(longer) >= len(shorter) + 2 and shorter in longer:
            return True
        n = min(6, len(key), len(prev))
        if n >= 5 and key[:n] == prev[:n]:
            return True
    return False


def merge_ocr_frames(
    frames: list[dict[str, Any]],
    min_len: int = 2,
) -> list[dict[str, Any]]:
    """跨帧去重：同一句字幕只保留首次出现的时间戳。"""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    canvas = _canvas_size(frames)
    for frame in frames:
        ts = frame.get("time")
        items = list(frame.get("texts") or [])
        in_buyer_show = any("买家秀" in str(x.get("text") or "") for x in items)
        for item in items:
            text = str(item.get("text") or "").strip()
            if len(text) < min_len:
                continue
            if in_buyer_show:
                # 评价弹层里的新字都丢掉；商品卡若仍露在边上，前面帧已经收过。
                continue
            if is_ocr_noise(text, box=item.get("box"), canvas=canvas):
                continue
            key = normalize_key(text)
            if not key or _is_near_duplicate(key, seen):
                continue
            seen.add(key)
            merged.append(
                {
                    "text": text,
                    "score": item.get("score"),
                    "time": ts,
                    "box": item.get("box"),
                }
            )
    return merged


def merge_record(
    sample_id: str,
    video_name: str,
    duration: float,
    asr: dict[str, Any] | None,
    ocr: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    asr = asr or {}
    ocr = ocr or {}
    asr_text = str(asr.get("text") or "").strip()
    ocr_lines = ocr.get("merged") or []
    # 最终结果是面向下游审核/分类的紧凑文本，不携带逐段时间戳和 OCR 框。
    ocr_text = " ".join(str(x.get("text") or "") for x in ocr_lines).strip()
    return {
        "audio_text": asr_text,
        "visual_text": ocr_text,
    }
