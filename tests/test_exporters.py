import pytest

from layr.agent import Agent
from layr.event import build_event
from layr.exporters.datadog import DatadogExporter
from layr.exporters.grafana import GrafanaExporter
from layr.exporters.layr_cloud import LayrCloudExporter
from layr.exporters.local import LocalExporter
from layr.session import new_session


@pytest.mark.asyncio
async def test_local_exporter_outputs(capsys) -> None:
    agent = Agent(api_key="k", exporter="local", mode="local")
    agent._exporter = LocalExporter()
    await agent.track(action="test", target="x", input_tokens=1, output_tokens=1, latency_ms=10)
    out = capsys.readouterr().out
    assert "[LAYR]" in out
    assert "action=test" in out
    await agent.aclose()


class DummyClient:
    def __init__(self) -> None:
        self.posts = 0

    async def post(self, *_args, **_kwargs):
        self.posts += 1

    async def aclose(self):
        return None


def _sample_event():
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
        actions_per_hour=1.0,
        cost_per_hour_usd=0.0,
        error_rate=0.0,
        deviation_from_baseline=0.0,
        metadata={},
        model=None,
    )


@pytest.mark.asyncio
async def test_http_exporters_can_export_and_close() -> None:
    event = _sample_event()
    dd = DatadogExporter(api_key="k")
    gr = GrafanaExporter(endpoint="http://g", token="t")
    lc = LayrCloudExporter(api_key="k", endpoint="http://l")
    dd._client = DummyClient()
    gr._client = DummyClient()
    lc._client = DummyClient()
    await dd.export([event])
    await gr.export([event])
    await lc.export([event])
    assert dd._client.posts == 1
    assert gr._client.posts == 1
    assert lc._client.posts == 1
    await dd.close()
    await gr.close()
    await lc.close()
