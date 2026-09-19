"""统一请求理解流水线。"""

from __future__ import annotations

from app.core.models import Dataset, RequestFrame, RequestResources
from app.models import ModelAdapter
from app.state import StateStore
from app.understanding.deterministic import extract_request_hints
from app.understanding.interpreter import RequestInterpreter, merge_deterministic_fields
from app.understanding.normalizer import RequestNormalizer
from app.understanding.reference_resolver import ReferenceResolver
from app.understanding.rule_gate import RuleGate
from app.understanding.state_snapshot import StateSnapshotLoader
from app.understanding.validator import RequestFrameValidator


class RequestUnderstandingPipeline:
    def __init__(
        self,
        store: StateStore,
        *,
        normalizer: RequestNormalizer | None = None,
        state_loader: StateSnapshotLoader | None = None,
        reference_resolver: ReferenceResolver | None = None,
        rule_gate: RuleGate | None = None,
        interpreter: RequestInterpreter | None = None,
        validator: RequestFrameValidator | None = None,
    ) -> None:
        self.normalizer = normalizer or RequestNormalizer()
        self.state_loader = state_loader or StateSnapshotLoader(store)
        self.reference_resolver = reference_resolver or ReferenceResolver(store)
        self.rule_gate = rule_gate or RuleGate()
        self.interpreter = interpreter or RequestInterpreter()
        self.validator = validator or RequestFrameValidator()

    async def understand(
        self,
        conversation_id: str,
        message: str,
        *,
        request_resources: RequestResources | None = None,
        datasets: list[Dataset] | None = None,
        model_adapter: ModelAdapter | None = None,
        exclude_run_id: str | None = None,
        exclude_task_id: str | None = None,
    ) -> RequestFrame:
        normalized = self.normalizer.normalize(message)
        state = self.state_loader.load(conversation_id, exclude_run_id=exclude_run_id, exclude_task_id=exclude_task_id)
        resolution = self.reference_resolver.resolve(normalized, state, request_resources)
        frame = self.rule_gate.match(normalized, state, resolution)
        if frame is None:
            frame = await self.interpreter.interpret(
                message=normalized,
                state=state,
                resolution=resolution,
                model_adapter=model_adapter,
                datasets=datasets,
            )
        frame = merge_deterministic_fields(frame, extract_request_hints(normalized, datasets or state.recent_datasets))
        merged = _merge_references(frame, resolution)
        return self.validator.validate(merged, state, request_resources)


def _merge_references(frame: RequestFrame, resolution) -> RequestFrame:
    references = list(resolution.references)
    seen = {(item.type, item.target_id) for item in references}
    for item in frame.references:
        if (item.type, item.target_id) not in seen:
            references.append(item)
            seen.add((item.type, item.target_id))
    return frame.model_copy(
        update={
            "references": references,
            "unresolved_references": list(dict.fromkeys([*resolution.unresolved_references, *frame.unresolved_references])),
        }
    )
