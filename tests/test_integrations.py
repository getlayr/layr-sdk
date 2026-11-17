import pytest

from layr.integrations.autogen import LayrConversableAgent
from layr.integrations.crewai import LayrCrew


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
