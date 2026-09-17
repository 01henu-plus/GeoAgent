"""请求理解专用的文本规整器。"""

from __future__ import annotations

import re


class RequestNormalizer:
    """只做输入清理，不猜测用户意图，也不改写业务语义。"""

    def normalize(self, message: str) -> str:
        return re.sub(r"\s+", " ", message.strip())

