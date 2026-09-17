"""GeoAgent 的 GIS 计划器。

计划不是前端展示文案，而是离线运行时真正执行的步骤清单。每个可执行步骤
都带有工具名和参数；缺少必要数据或参数时，计划会明确进入询问用户状态，
不会用第一个数据集或隐含默认值替用户做关键选择。
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.models import Dataset, DatasetKind, IntentResult, IntentType, Plan, PlanStep


class Planner:
    """把结构化意图编译成可执行的 GIS Plan。"""

    def build(self, goal: str, intent: IntentResult, datasets: list[Dataset]) -> Plan:
        entities = intent.entities
        operations = [str(item) for item in entities.get("operations", []) if item]
        operation = operations[0] if operations else str(entities.get("operation") or "")
        selected = _ordered_selected(datasets, entities.get("mentioned_dataset_ids"), entities.get("dataset_roles"))
        metadata = {
            "operation": operation or None,
            "operation_sequence": operations,
            "selected_dataset_ids": [item.id for item in selected],
            "dataset_count": len(selected),
        }

        if intent.intent in {IntentType.UNKNOWN, IntentType.KNOWLEDGE_QUERY, IntentType.RESULT_INTERPRETATION, IntentType.RUN_DIAGNOSIS}:
            return Plan(goal=goal, intent=intent.intent, metadata=metadata)

        if self._should_delegate(intent, entities, selected):
            roles = [role for role in ("road", "population", "terrain") if entities.get(f"{role}_requested")]
            steps = [
                PlanStep(id="decompose", title="拆分主题任务", action="decompose", description=f"按 {'、'.join(roles)} 分配独立空间分析", tool_name=None),
                PlanStep(id="parallel", title="并行执行主题分析", action="delegate", depends_on=["decompose"], description="为每个独立主题创建一个临时智能体", tool_name=None),
                PlanStep(id="synthesize", title="验证并汇总主题结果", action="synthesize", depends_on=["parallel"], description="检查子任务结果并形成统一结论", tool_name=None),
            ]
            metadata["delegated_roles"] = roles
            return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

        if not selected and entities.get("requires_dataset"):
            return Plan(goal=goal, intent=intent.intent, clarification=_missing_dataset_message(operation), metadata=metadata)

        if entities.get("render_requested") and not operations:
            return self._render_plan(goal, intent, selected, metadata)

        if len(operations) > 1:
            composed = self._composed_plan(goal, intent, selected, operations, metadata)
            if composed is not None:
                return composed

        if intent.intent is IntentType.DATA_INSPECTION and not operation:
            steps = _inspection_steps(selected)
            if not steps:
                return Plan(goal=goal, intent=intent.intent, clarification="请先添加或登记要检查的空间数据。", metadata=metadata)
            return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

        if operation == "buffer":
            return self._buffer_plan(goal, intent, selected, metadata)
        if operation == "distance":
            return self._distance_plan(goal, intent, selected, metadata)
        if operation == "clip" and _first_kind(selected, DatasetKind.RASTER) is not None and _first_kind(selected, DatasetKind.VECTOR) is not None:
            return self._raster_clip_plan(goal, intent, selected, metadata)
        if operation in {"clip", "intersection", "spatial_join"}:
            return self._binary_vector_plan(goal, intent, selected, operation, metadata)
        if operation == "zonal_statistics":
            return self._zonal_plan(goal, intent, selected, metadata)
        if operation == "slope":
            return self._slope_plan(goal, intent, selected, metadata)
        if operation == "reproject":
            return self._reproject_plan(goal, intent, selected, metadata)
        if operation in {"dissolve", "repair", "validate"}:
            return self._single_vector_plan(goal, intent, selected, operation, metadata)

        return Plan(
            goal=goal,
            intent=intent.intent,
            steps=[PlanStep(id="understand", title="整理任务上下文", action="context.prepare", description="等待模型或后续决策器补充具体操作")],
            metadata=metadata,
        )

    def _render_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], metadata: dict[str, object]) -> Plan:
        dataset = _first_kind(datasets, DatasetKind.VECTOR)
        if dataset is None:
            return Plan(goal=goal, intent=intent.intent, clarification="当前地图发布工具需要一个矢量数据集，请先添加矢量图层。", metadata=metadata)
        steps = _inspection_steps([dataset])
        steps.append(
            PlanStep(
                id="publish",
                title="生成可查看地图",
                action="map.render",
                tool_name="map.render",
                arguments={"dataset_id": dataset.id, "title": dataset.name},
                depends_on=[steps[-1].id],
            )
        )
        metadata.update({"operation": "render", "output_policy": "artifact"})
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _raster_clip_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], metadata: dict[str, object]) -> Plan:
        raster = _first_kind(datasets, DatasetKind.RASTER)
        mask = _first_kind(datasets, DatasetKind.VECTOR)
        if raster is None or mask is None:
            return Plan(goal=goal, intent=intent.intent, clarification="栅格裁剪需要一个栅格数据集和一个矢量边界。", metadata=metadata)
        steps = _inspection_steps([raster, mask])
        steps.append(
            PlanStep(
                id="clip",
                title="裁剪栅格数据",
                action="raster.clip",
                tool_name="raster.clip",
                arguments={"dataset_id": raster.id, "mask_dataset_id": mask.id},
                depends_on=[item.id for item in steps],
            )
        )
        metadata.update({"operation": "clip", "output_policy": "derived_dataset"})
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _composed_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], operations: list[str], metadata: dict[str, object]) -> Plan | None:
        """把常见前置处理和空间分析编译成一条真正可执行的链。"""

        primary = next((item for item in reversed(operations) if item not in {"reproject", "repair", "validate"}), None)
        target_crs = intent.entities.get("target_crs") or "auto"
        if primary == "buffer" and "reproject" in operations:
            vector = _first_kind(datasets, DatasetKind.VECTOR)
            distance = intent.entities.get("distance")
            if vector is None:
                return Plan(goal=goal, intent=intent.intent, clarification="重投影并生成缓冲区需要一个矢量数据集。", metadata=metadata)
            if not isinstance(distance, (int, float)) or isinstance(distance, bool) or float(distance) <= 0:
                return Plan(goal=goal, intent=intent.intent, clarification="请告诉我要生成多大距离的缓冲区，例如“生成 500 米缓冲区”。", metadata=metadata)
            steps = _inspection_steps([vector])
            steps.extend([
                PlanStep(id="reproject", title="准备平面坐标系", action="crs.reproject", tool_name="crs.reproject", arguments={"dataset_id": vector.id, "target_crs": target_crs}, depends_on=["inspect"]),
                PlanStep(id="buffer", title="生成缓冲区", action="vector.buffer", tool_name="vector.buffer", arguments={"dataset_id": "${reproject.dataset_id}", "distance": float(distance)}, depends_on=["reproject"]),
                PlanStep(id="verify", title="验证缓冲区结果", action="vector.validate", tool_name="vector.validate", arguments={"dataset_id": "${buffer.dataset_id}"}, depends_on=["buffer"]),
                PlanStep(id="publish", title="生成可查看地图", action="map.render", tool_name="map.render", arguments={"dataset_id": "${buffer.dataset_id}", "title": f"{vector.name} 缓冲区"}, depends_on=["verify"]),
            ])
            metadata.update({"operation": "buffer", "output_policy": "dataset_and_map", "distance": float(distance)})
            return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

        if primary == "slope" and "reproject" in operations:
            raster = _first_kind(datasets, DatasetKind.RASTER)
            if raster is None:
                return Plan(goal=goal, intent=intent.intent, clarification="重投影并计算坡度需要一个栅格 DEM。", metadata=metadata)
            steps = _inspection_steps([raster])
            steps.extend([
                PlanStep(id="reproject", title="准备 DEM 坐标系", action="raster.reproject", tool_name="raster.reproject", arguments={"dataset_id": raster.id, "target_crs": target_crs}, depends_on=["inspect"]),
                PlanStep(id="slope", title="计算坡度栅格", action="raster.slope", tool_name="raster.slope", arguments={"dataset_id": "${reproject.dataset_id}"}, depends_on=["reproject"]),
            ])
            metadata.update({"operation": "slope", "output_policy": "derived_dataset"})
            return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

        if primary in {"clip", "intersection", "spatial_join"} and "reproject" in operations:
            vectors = [item for item in datasets if item.kind is DatasetKind.VECTOR]
            if len(vectors) < 2:
                return Plan(goal=goal, intent=intent.intent, clarification=f"重投影并{_operation_label(primary)}需要两个矢量数据集。", metadata=metadata)
            left, right = vectors[:2]
            steps = _inspection_steps([left, right])
            steps.extend([
                PlanStep(id="reproject_left", title=f"准备 {left.name} 坐标系", action="crs.reproject", tool_name="crs.reproject", arguments={"dataset_id": left.id, "target_crs": target_crs}, depends_on=["inspect", "inspect_2"]),
                PlanStep(id="reproject_right", title=f"准备 {right.name} 坐标系", action="crs.reproject", tool_name="crs.reproject", arguments={"dataset_id": right.id, "target_crs": target_crs}, depends_on=["inspect", "inspect_2"]),
            ])
            if primary == "clip":
                arguments = {"dataset_id": "${reproject_left.dataset_id}", "mask_dataset_id": "${reproject_right.dataset_id}"}
            elif primary == "spatial_join":
                arguments = {"left_dataset_id": "${reproject_left.dataset_id}", "right_dataset_id": "${reproject_right.dataset_id}", "predicate": intent.entities.get("predicate", "intersects")}
            else:
                arguments = {"left_dataset_id": "${reproject_left.dataset_id}", "right_dataset_id": "${reproject_right.dataset_id}"}
            steps.extend([
                PlanStep(id="operate", title=_operation_label(primary), action=f"vector.{primary}", tool_name=f"vector.{primary}", arguments=arguments, depends_on=["reproject_left", "reproject_right"]),
                PlanStep(id="verify", title="验证空间结果", action="vector.validate", tool_name="vector.validate", arguments={"dataset_id": "${operate.dataset_id}"}, depends_on=["operate"]),
            ])
            metadata.update({"operation": primary, "output_policy": "derived_dataset"})
            return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

        if primary == "buffer" and "repair" in operations:
            vector = _first_kind(datasets, DatasetKind.VECTOR)
            distance = intent.entities.get("distance")
            if vector is None or not isinstance(distance, (int, float)) or isinstance(distance, bool) or float(distance) <= 0:
                return Plan(goal=goal, intent=intent.intent, clarification="修复后生成缓冲区需要一个矢量数据集和正的缓冲距离。", metadata=metadata)
            steps = _inspection_steps([vector])
            steps.extend([
                PlanStep(id="repair", title="修复矢量几何", action="vector.repair", tool_name="vector.repair", arguments={"dataset_id": vector.id}, depends_on=["inspect"]),
                PlanStep(id="buffer", title="生成缓冲区", action="vector.buffer", tool_name="vector.buffer", arguments={"dataset_id": "${repair.dataset_id}", "distance": float(distance)}, depends_on=["repair"]),
                PlanStep(id="verify", title="验证缓冲区结果", action="vector.validate", tool_name="vector.validate", arguments={"dataset_id": "${buffer.dataset_id}"}, depends_on=["buffer"]),
            ])
            metadata.update({"operation": "buffer", "output_policy": "derived_dataset", "distance": float(distance)})
            return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)
        return None

    @staticmethod
    def _should_delegate(intent: IntentResult, entities: dict[str, object], datasets: list[Dataset]) -> bool:
        roles = sum(bool(entities.get(key)) for key in ("road_requested", "population_requested", "terrain_requested"))
        return intent.intent is IntentType.SPATIAL_ANALYSIS and roles >= 2 and len(datasets) >= 2

    def _buffer_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], metadata: dict[str, object]) -> Plan:
        vector = _first_kind(datasets, DatasetKind.VECTOR)
        distance = intent.entities.get("distance")
        if vector is None:
            return Plan(goal=goal, intent=intent.intent, clarification="缓冲区需要一个矢量数据集，请先添加道路、边界或其他矢量图层。", metadata=metadata)
        if not isinstance(distance, (int, float)) or isinstance(distance, bool) or float(distance) <= 0:
            return Plan(goal=goal, intent=intent.intent, clarification="请告诉我要生成多大距离的缓冲区，例如“生成 500 米缓冲区”。", metadata=metadata)
        steps = _inspection_steps([vector])
        steps.extend(
            [
                PlanStep(id="buffer", title="生成缓冲区", action="vector.buffer", tool_name="vector.buffer", arguments={"dataset_id": vector.id, "distance": float(distance)}, depends_on=["inspect"]),
                PlanStep(id="verify", title="验证缓冲区结果", action="vector.validate", tool_name="vector.validate", arguments={"dataset_id": "${buffer.dataset_id}"}, depends_on=["buffer"]),
                PlanStep(id="publish", title="生成可查看地图", action="map.render", tool_name="map.render", arguments={"dataset_id": "${buffer.dataset_id}", "title": f"{vector.name} 缓冲区"}, depends_on=["verify"]),
            ]
        )
        metadata["output_policy"] = "dataset_and_map"
        metadata["distance"] = float(distance)
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _distance_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], metadata: dict[str, object]) -> Plan:
        vectors = [item for item in datasets if item.kind is DatasetKind.VECTOR]
        if len(vectors) < 2:
            return Plan(goal=goal, intent=intent.intent, clarification="距离分析需要两个矢量数据集，请同时选择源数据和目标数据。", metadata=metadata)
        source, target = vectors[:2]
        steps = _inspection_steps([source, target])
        args: dict[str, object] = {"source_dataset_id": source.id, "target_dataset_id": target.id}
        threshold = intent.entities.get("distance")
        if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
            args["threshold"] = float(threshold)
        steps.append(PlanStep(id="analyze", title="计算距离分布", action="analysis.distance", tool_name="analysis.distance", arguments=args, depends_on=[item.id for item in steps]))
        metadata["output_policy"] = "analysis_summary"
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _binary_vector_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], operation: str, metadata: dict[str, object]) -> Plan:
        vectors = [item for item in datasets if item.kind is DatasetKind.VECTOR]
        if len(vectors) < 2:
            return Plan(goal=goal, intent=intent.intent, clarification=f"{_operation_label(operation)}需要两个矢量数据集，请选择数据主体和边界/关联图层。", metadata=metadata)
        left, right = vectors[:2]
        steps = _inspection_steps([left, right])
        if operation == "clip":
            arguments = {"dataset_id": left.id, "mask_dataset_id": right.id}
        elif operation == "spatial_join":
            arguments = {"left_dataset_id": left.id, "right_dataset_id": right.id, "predicate": intent.entities.get("predicate", "intersects")}
        else:
            arguments = {"left_dataset_id": left.id, "right_dataset_id": right.id}
        steps.append(PlanStep(id="operate", title=_operation_label(operation), action=f"vector.{operation}", tool_name=f"vector.{operation}", arguments=arguments, depends_on=[item.id for item in steps]))
        steps.append(PlanStep(id="verify", title="验证空间结果", action="vector.validate", tool_name="vector.validate", arguments={"dataset_id": "${operate.dataset_id}"}, depends_on=["operate"]))
        steps.append(PlanStep(id="publish", title="生成可查看地图", action="map.render", tool_name="map.render", arguments={"dataset_id": "${operate.dataset_id}", "title": f"{left.name} {_operation_label(operation)}"}, depends_on=["verify"]))
        metadata["output_policy"] = "dataset_and_map"
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _zonal_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], metadata: dict[str, object]) -> Plan:
        zones = _first_kind(datasets, DatasetKind.VECTOR)
        raster = _first_kind(datasets, DatasetKind.RASTER)
        if zones is None or raster is None:
            return Plan(goal=goal, intent=intent.intent, clarification="分区统计需要一个矢量分区图层和一个栅格数据集（例如 DEM）。", metadata=metadata)
        steps = _inspection_steps([zones, raster])
        steps.append(PlanStep(id="analyze", title="计算分区统计", action="analysis.zonal_statistics", tool_name="analysis.zonal_statistics", arguments={"zones_dataset_id": zones.id, "raster_dataset_id": raster.id}, depends_on=[item.id for item in steps]))
        metadata["output_policy"] = "analysis_summary"
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _slope_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], metadata: dict[str, object]) -> Plan:
        raster = _first_kind(datasets, DatasetKind.RASTER)
        if raster is None:
            return Plan(goal=goal, intent=intent.intent, clarification="坡度分析需要一个栅格 DEM，请先添加或选择 DEM 文件。", metadata=metadata)
        steps = _inspection_steps([raster])
        steps.append(PlanStep(id="slope", title="计算坡度栅格", action="raster.slope", tool_name="raster.slope", arguments={"dataset_id": raster.id}, depends_on=["inspect"]))
        metadata["output_policy"] = "derived_dataset"
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _reproject_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], metadata: dict[str, object]) -> Plan:
        if not datasets:
            return Plan(goal=goal, intent=intent.intent, clarification="重投影需要一个输入数据集。", metadata=metadata)
        dataset = datasets[0]
        target_crs = intent.entities.get("target_crs") or "auto"
        steps = _inspection_steps([dataset])
        tool_name = "raster.reproject" if dataset.kind is DatasetKind.RASTER else "crs.reproject"
        steps.append(PlanStep(id="reproject", title="执行重投影", action=tool_name, tool_name=tool_name, arguments={"dataset_id": dataset.id, "target_crs": target_crs}, depends_on=["inspect"]))
        metadata["output_policy"] = "derived_dataset"
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)

    def _single_vector_plan(self, goal: str, intent: IntentResult, datasets: list[Dataset], operation: str, metadata: dict[str, object]) -> Plan:
        vector = _first_kind(datasets, DatasetKind.VECTOR)
        if vector is None:
            return Plan(goal=goal, intent=intent.intent, clarification=f"{_operation_label(operation)}需要一个矢量数据集。", metadata=metadata)
        steps = _inspection_steps([vector])
        if operation == "dissolve":
            arguments = {"dataset_id": vector.id}
            field = intent.entities.get("field")
            if isinstance(field, str) and field:
                arguments["by"] = field
        else:
            arguments = {"dataset_id": vector.id}
        tool_name = "vector." + operation if operation != "validate" else "vector.validate"
        steps.append(PlanStep(id="operate", title=_operation_label(operation), action=tool_name, tool_name=tool_name, arguments=arguments, depends_on=["inspect"]))
        if operation not in {"validate"}:
            steps.append(PlanStep(id="verify", title="验证空间结果", action="vector.validate", tool_name="vector.validate", arguments={"dataset_id": "${operate.dataset_id}"}, depends_on=["operate"]))
        metadata["output_policy"] = "derived_dataset" if operation != "validate" else "analysis_summary"
        return Plan(goal=goal, intent=intent.intent, steps=steps, metadata=metadata)


def _inspection_steps(datasets: Iterable[Dataset]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for index, dataset in enumerate(datasets):
        step_id = "inspect" if index == 0 else f"inspect_{index + 1}"
        steps.append(PlanStep(id=step_id, title=f"检查数据：{dataset.name}", action="dataset.inspect", tool_name="dataset.inspect", arguments={"dataset_id": dataset.id}, description="读取 CRS、范围、字段和轻量统计"))
    return steps


def _ordered_selected(datasets: list[Dataset], mentioned: object, roles: object) -> list[Dataset]:
    ids = {str(value) for value in mentioned} if isinstance(mentioned, list) else set()
    role_values = {str(value) for value in roles} if isinstance(roles, list) else set()
    if ids:
        selected = [item for item in datasets if item.id in ids]
        if selected:
            return selected
    if role_values:
        matched: list[Dataset] = []
        for item in datasets:
            text = f"{item.name} {item.path}".casefold()
            if "road" in role_values and any(word in text for word in ("road", "道路", "路网")):
                matched.append(item)
            elif "population" in role_values and any(word in text for word in ("population", "人口", "pop")):
                matched.append(item)
            elif "terrain" in role_values and any(word in text for word in ("dem", "terrain", "elevation", "高程", "地形")):
                matched.append(item)
            elif "boundary" in role_values and any(word in text for word in ("boundary", "mask", "边界", "行政区")):
                matched.append(item)
        if matched:
            return matched
    return list(datasets)


def _first_kind(datasets: list[Dataset], kind: DatasetKind) -> Dataset | None:
    return next((item for item in datasets if item.kind is kind), None)


def _missing_dataset_message(operation: str) -> str:
    return f"{_operation_label(operation) if operation else '这个操作'}需要输入数据，请先添加或登记空间数据文件。"


def _operation_label(operation: str) -> str:
    return {
        "buffer": "生成缓冲区",
        "clip": "裁剪数据",
        "intersection": "计算相交区域",
        "spatial_join": "执行空间连接",
        "dissolve": "融合矢量要素",
        "zonal_statistics": "计算分区统计",
        "slope": "计算坡度",
        "reproject": "重投影",
        "repair": "修复矢量几何",
        "validate": "验证矢量几何",
        "distance": "进行距离分析",
    }.get(operation, operation or "空间分析")


__all__ = ["Planner"]
