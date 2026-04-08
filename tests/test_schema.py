from layr.schema import LayrEvent


def test_schema_to_dict() -> None:
    e = LayrEvent(
        agent_id="a",
        agent_name="n",
        agent_version="0.0.1",
        framework="custom",
        environment="dev",
        model="gpt-4o",
        session_id="s",
        session_start="t0",
        triggered_by="u",
        action_type="x",
        action_target="y",
        action_duration_ms=1,
        action_success=True,
        action_error="",
        intent="",
        reasoning="",
        confidence=1.0,
        tools_considered=[],
        tools_used=[],
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        estimated_cost_usd=0.1,
        model_latency_ms=1,
        prompt_version="v",
        parent_agent_id="",
        handoff_count=0,
        delegation_depth=0,
        network_id="",
        actions_per_hour=1.0,
        cost_per_hour_usd=0.1,
        error_rate=0.0,
        deviation_from_baseline=0.0,
        timestamp="now",
        layr_sdk_version="0.1.1",
        metadata={"k": "v"},
    )
    d = e.to_dict()
    assert d["agent_id"] == "a"
    assert d["metadata"]["k"] == "v"
