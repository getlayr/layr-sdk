import pytest

from layr.event import build_event
from layr.session import new_session


def test_event_confidence_is_clamped():
    ev = build_event(
        agent_id="a",
        agent_name="n",
        agent_version="0.0.1",
        framework="custom",
        environment="dev",
        default_model="gpt-4o",
        session=new_session("u"),
        action_type="x",
        action_target="y",
        action_duration_ms=5,
        action_success=True,
        action_error="",
        intent="",
        reasoning="",
        confidence=2.0,
        tools_considered=[],
        tools_used=[],
        input_tokens=1,
        output_tokens=1,
        prompt_version="v1",
        parent_agent_id="",
        handoff_count=0,
        delegation_depth=0,
        network_id="",
        actions_per_hour=1.0,
        cost_per_hour_usd=0.0,
        error_rate=0.0,
        deviation_from_baseline=0.0,
        metadata={},
        model=None,
    )
    assert ev.confidence == 1.0
