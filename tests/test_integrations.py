import pytest

from layr.integrations.autogen import LayrConversableAgent
from layr.integrations.crewai import LayrCrew
from layr.integrations.langchain import LayrCallbackHandler


@pytest.mark.asyncio
async def test_langchain_handler_tracks(monkeypatch) -> None:
    monkeypatch.setenv("LAYR_EXPORTER", "local")
    monkeypatch.setenv("LAYR_MODE", "local")
    h = LayrCallbackHandler(api_key="k", agent_name="lc", environment="dev")
    await h.on_chain_start({"name": "chain"}, {"q": "hello"})
    await h.on_llm_end(response="ok", model="gpt-4o", input_tokens=10, output_tokens=5)
    await h.on_tool_end("tool output", tool_name="search", latency_ms=12)
    await h.on_chain_error(RuntimeError("x"))
    await h._agent.aclose()


class MockCrew:
    def kickoff(self):
        return "done"


@pytest.mark.asyncio
async def test_crewai_and_autogen_wrappers(monkeypatch) -> None:
    monkeypatch.setenv("LAYR_EXPORTER", "local")
    monkeypatch.setenv("LAYR_MODE", "local")
    crew = LayrCrew(api_key="k", crew=MockCrew(), environment="dev")
    res = await crew.kickoff()
    assert res == "done"
    await crew._agent.aclose()

    ag = LayrConversableAgent(api_key="k", name="assistant", environment="dev")
    await ag.send("hello")
    await ag.receive("world")
    await ag._agent.aclose()
