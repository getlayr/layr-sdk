from layr.anomaly import Baselines, compute_deviation, is_anomalous
from layr.event import build_event
from layr.session import new_session


def _event(actions: float, cost: float, err: float):
    return build_event(
        agent_id="a",
        agent_name="n",
        agent_version="0.0.1",
        framework="custom",
        environment="dev",
        default_model="gpt-4o",
        session=new_session("u"),
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
        prompt_version="v",
        parent_agent_id="",
        handoff_count=0,
        delegation_depth=0,
        network_id="",
        actions_per_hour=actions,
        cost_per_hour_usd=cost,
        error_rate=err,
        deviation_from_baseline=0.0,
        metadata={},
        model=None,
    )


def test_anomaly_detection_math():
    b = Baselines(actions_per_hour=10, cost_per_hour_usd=1.0, error_rate=0.1)
    normal = _event(10, 1.0, 0.1)
    spike = _event(50, 5.0, 0.5)
    assert compute_deviation(normal, b) < compute_deviation(spike, b)
    assert not is_anomalous(normal, b, threshold=2.0)
    assert is_anomalous(spike, b, threshold=1.0)
