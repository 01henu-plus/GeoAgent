"""只从明确的长期偏好表达中提取 UserProfile 更新。"""

from __future__ import annotations

import re

from app.core.models import MeasurementSystem, ResponseStyle


class ProfilePreferenceExtractor:
    MARKERS = ("以后", "默认", "一直", "之后都", "from now on", "always", "default")
    SENSITIVE = ("密码", "password", "api_key", "api key", "token", "session", "authorization", "bearer")

    def extract(self, text: str) -> dict[str, str]:
        normalized = text.strip()
        folded = normalized.casefold()
        if not normalized or any(item in folded for item in self.SENSITIVE):
            return {}
        if not any(marker.casefold() in folded for marker in self.MARKERS):
            return {}
        changes: dict[str, str] = {}
        if re.search(r"(?:以后|默认|一直|之后都|from now on|always|default).*?(?:中文|chinese)", folded):
            changes["language"] = "zh-CN"
        elif re.search(r"(?:以后|默认|一直|之后都|from now on|always|default).*?(?:英文|english)", folded):
            changes["language"] = "en-US"
        if re.search(r"(?:以后|默认|一直|之后都|from now on|always|default).*?(?:简洁|简短|concise)", folded):
            changes["response_style"] = ResponseStyle.CONCISE.value
        elif re.search(r"(?:以后|默认|一直|之后都|from now on|always|default).*?(?:详细|详细解释|detailed)", folded):
            changes["response_style"] = ResponseStyle.DETAILED.value
        if re.search(r"(?:默认|以后|一直|之后都|from now on|always|default).*?(?:公制|metric)", folded):
            changes["measurement_system"] = MeasurementSystem.METRIC.value
        elif re.search(r"(?:默认|以后|一直|之后都|from now on|always|default).*?(?:英制|imperial)", folded):
            changes["measurement_system"] = MeasurementSystem.IMPERIAL.value
        for label, value in (("GeoPackage", "GeoPackage"), ("GeoJSON", "GeoJSON"), ("GeoTIFF", "GeoTIFF"), ("CSV", "CSV")):
            if re.search(rf"(?:默认|以后|一直|之后都|from now on|always|default).*?{re.escape(label.casefold())}", folded):
                changes["preferred_output_format"] = value
                break
        return changes


__all__ = ["ProfilePreferenceExtractor"]
