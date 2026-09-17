"""最终回答的稳定格式。"""

from __future__ import annotations

from app.core.models import AgentResult, AgentResultStatus


def summarize(result: AgentResult) -> str:
    status = "完成" if result.status is AgentResultStatus.SUCCESS else "部分完成" if result.status is AgentResultStatus.PARTIAL else "未完成"
    lines = [f"{status}：{result.summary}"]
    if result.findings:
        lines.append("\n关键发现：")
        lines.extend(f"- {finding}" for finding in result.findings[:8])
    if result.datasets:
        lines.append(f"\n关联数据集：{', '.join(result.datasets)}")
    if result.artifacts:
        lines.append(f"交付产物：{', '.join(result.artifacts)}")
    if result.warnings:
        lines.append("\n注意：" + "；".join(result.warnings[:5]))
    return "\n".join(lines)

