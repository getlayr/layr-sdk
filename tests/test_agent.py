import pytest

from layr.agent import Agent, AgentNetwork
from layr.exporters.local import LocalExporter


@pytest.mark.asyncio
async def test_track_local_mode_builds_event():
    agent = Agent(
        api_key="key",
        exporter="local",
        mode="local",
        fail_silent=False,
    )
    event = await agent.track(action="send_email", target="a@b.com", input_tokens=10, output_tokens=20)
    assert event.total_tokens == 30
    assert event.action_type == "send_email"
    await agent.aclose()


@pytest.mark.asyncio
async def test_agent_network_handoff(monkeypatch):
    monkeypatch.setenv("LAYR_EXPORTER", "local")
    monkeypatch.setenv("LAYR_MODE", "local")
    net = AgentNetwork(api_key="k")
    a = net.agent("orchestrator")
    b = net.agent("worker")
    await net.track_handoff(from_agent=a, to_agent=b, task="collect_data")
    assert b._parent_agent_id == a.agent_id
    await a.aclose()
    await b.aclose()


@pytest.mark.asyncio
async def test_session_context_and_anomaly_callback():
    seen = {"count": 0}

    def on_anomaly(_event):
        seen["count"] += 1

    agent = Agent(
        api_key="k",
        exporter="local",
        mode="local",
        anomaly_detection=True,
        baselines={"actions_per_hour": 0.1, "cost_per_hour_usd": 0.000001, "error_rate": 0.0},
        on_anomaly=on_anomaly,
    )
    with agent.session(triggered_by="user_1"):
        event = await agent.track(action="x", target="y", success=False)
    assert event.triggered_by == "user_1"
    assert seen["count"] >= 1
    await agent.aclose()


@pytest.mark.asyncio
async def test_send_with_retry_non_silent():
    class FailingExporter(LocalExporter):
        def __init__(self):
            self.calls = 0

        async def export(self, events):
            self.calls += 1
            if self.calls < 2:
                raise RuntimeError("fail")
            return await super().export(events)

    agent = Agent(api_key="k", exporter="local", mode="production", fail_silent=False)
    agent._exporter = FailingExporter()
    await agent.track(action="x")
    await agent.flush()
    assert agent._exporter.calls >= 2
    await agent.aclose()
