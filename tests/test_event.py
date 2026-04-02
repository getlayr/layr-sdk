from layr.event import build_event
from layr.session import new_session


def test_event_total_tokens_and_cost() -> None:
    event = build_event(
        agent_id="agent",
        agent_name="agent",
        agent_version="0.0.1",
        framework="custom",
        environment="dev",
        default_model="gpt-4o",
        session=new_session("user"),
        action_type="send_email",
        action_target="target",
        action_duration_ms=100,
        action_success=True,
        action_error="",
        intent="intent",
        reasoning="reasoning",
        confidence=0.5,
        tools_considered=["a"],
        tools_used=["a"],
        input_tokens=100,
        output_tokens=50,
        prompt_version="v1",
        parent_agent_id="",
        handoff_count=0,
        delegation_depth=0,
        network_id="",
        actions_per_hour=10.0,
        cost_per_hour_usd=0.0,
        error_rate=0.0,
        deviation_from_baseline=0.0,
        metadata={},
        model="gpt-4o",
    )
    assert event.total_tokens == 150
    assert event.estimated_cost_usd > 0
