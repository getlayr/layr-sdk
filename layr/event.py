from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from layr.schema import LayrEvent
from layr.session import SessionState
from layr.utils import SDK_VERSION, estimate_cost_usd, utc_now_iso


def build_event(
    *,
    agent_id: str,
    agent_name: str,
    agent_version: str,
    framework: str,
    environment: str,
    default_model: str,
    session: SessionState,
    action_type: str,
    action_target: str,
    action_duration_ms: int,
    action_success: bool,
    action_error: str,
    intent: str,
    reasoning: str,
    confidence: float,
    tools_considered: Optional[List[str]],
    tools_used: Optional[List[str]],
    input_tokens: int,
    output_tokens: int,
    prompt_version: str,
    parent_agent_id: str,
    handoff_count: int,
    delegation_depth: int,
    network_id: str,
    actions_per_hour: float,
    cost_per_hour_usd: float,
    error_rate: float,
    deviation_from_baseline: float,
    metadata: Optional[Mapping[str, object]],
    model: Optional[str],
    timestamp: Optional[str] = None,
) -> LayrEvent:
    m = model or default_model
    in_t = max(0, int(input_tokens))
    out_t = max(0, int(output_tokens))
    total = in_t + out_t
    conf = max(0.0, min(1.0, float(confidence)))
    cost = estimate_cost_usd(m, in_t, out_t)

    event = LayrEvent(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_version=agent_version,
        framework=framework,
        environment=environment,
        model=m,
        session_id=session.session_id,
        session_start=session.session_start,
        triggered_by=session.triggered_by,
        action_type=action_type,
        action_target=action_target,
        action_duration_ms=max(0, int(action_duration_ms)),
        action_success=bool(action_success),
        action_error=action_error,
        intent=intent,
        reasoning=reasoning,
        confidence=conf,
        tools_considered=list(tools_considered or []),
        tools_used=list(tools_used or []),
        input_tokens=in_t,
        output_tokens=out_t,
        total_tokens=total,
        estimated_cost_usd=cost,
        model_latency_ms=max(0, int(action_duration_ms)),
        prompt_version=prompt_version,
        parent_agent_id=parent_agent_id,
        handoff_count=max(0, int(handoff_count)),
        delegation_depth=max(0, int(delegation_depth)),
        network_id=network_id,
        actions_per_hour=float(actions_per_hour),
        cost_per_hour_usd=float(cost_per_hour_usd),
        error_rate=float(error_rate),
        deviation_from_baseline=float(deviation_from_baseline),
        timestamp=timestamp or utc_now_iso(),
        layr_sdk_version=SDK_VERSION,
        metadata=dict(metadata or {}),
    )
    _promote_evidence_fields(event)
    return event


def _md_str(metadata: Dict[str, object], key: str) -> str:
    value = metadata.get(key, "")
    if isinstance(value, str):
        return value
    return ""


def _promote_evidence_fields(event: LayrEvent) -> None:
    """Copy hot evidence keys out of metadata so exporters see first-class fields."""
    metadata = event.metadata
    event.evidence_kind = _md_str(metadata, "evidence_kind")
    event.decision_id = _md_str(metadata, "decision_id")
    event.prev_hash = _md_str(metadata, "prev_hash")
    event.evidence_hash = _md_str(metadata, "evidence_hash")
    event.article_26 = _md_str(metadata, "article_26")
    event.article_26_duty = _md_str(metadata, "article_26_duty")
