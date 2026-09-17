"""请求交互模式的低层词法判断，不包含状态或路由策略。"""

from __future__ import annotations

import re

_CANCEL_RE = re.compile(r"^(停止|取消|算了|结束任务|终止)(这个任务|当前任务|任务)?[。！!、，,]?$", re.IGNORECASE)
_CONTINUE_RE = re.compile(r"^(继续|接着做|继续刚才的|继续上一次|沿用刚才的)(吧|做|任务|分析)?[。！!，,]?$", re.IGNORECASE)
_RETRY_RE = re.compile(r"^(再试一次|重试|重新来|重新执行|再跑一次)[。！!，,]?$", re.IGNORECASE)


def is_cancel_request(text: str, *, strict: bool = False) -> bool:
    value = text.strip()
    return bool(_CANCEL_RE.fullmatch(value)) if strict else value.casefold().startswith(("停止", "取消", "算了", "结束任务", "终止"))


def is_continue_request(text: str, *, strict: bool = False) -> bool:
    value = text.strip()
    if strict:
        return bool(_CONTINUE_RE.fullmatch(value))
    lowered = value.casefold()
    return bool(_CONTINUE_RE.fullmatch(value)) or lowered == "继续" or lowered.startswith(("继续", "接着", "沿用刚才")) or (lowered.endswith(("继续", "接着")) and any(term in lowered for term in ("用", "数据", "结果", "这个", "它", "刚才")))


def is_retry_request(text: str, *, strict: bool = False) -> bool:
    value = text.strip()
    return bool(_RETRY_RE.fullmatch(value)) if strict else value.casefold().startswith(("再试一次", "重试", "重新来", "重新执行", "再跑一次"))


def is_modify_request(text: str) -> bool:
    value = text.strip()
    return value.startswith(("不对", "把", "改成", "换成", "调整")) and any(term in value for term in ("改", "换", "调整", "范围", "条件", "参数"))


def strip_modify_prefix(message: str) -> str:
    return re.sub(r"^(不对[，, ]*|把|改成|换成|调整[为成]?)[ ]*", "", message, flags=re.IGNORECASE).strip("，,。；; ")


__all__ = ["is_cancel_request", "is_continue_request", "is_modify_request", "is_retry_request", "strip_modify_prefix"]
