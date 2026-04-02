from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class LayrEvent:
    agent_id: str
    agent_name: str
    agent_version: str
    framework: str
    environment: str
    model: str
    session_id: str
    session_start: str
    triggered_by: str
    action_type: str
    action_target: str
    action_duration_ms: int
    action_success: bool
    action_error: str
    intent: str
    reasoning: str
    confidence: float
    tools_considered: List[str]
    tools_used: List[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    model_latency_ms: int
    prompt_version: str
    parent_agent_id: str
    handoff_count: int
    delegation_depth: int
    network_id: str
    actions_per_hour: float
    cost_per_hour_usd: float
    error_rate: float
    deviation_from_baseline: float
    timestamp: str
    layr_sdk_version: str
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
