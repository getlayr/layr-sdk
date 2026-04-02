from __future__ import annotations

from typing import Any

from layr.agent import Agent


class LayrCrew:
    def __init__(self, api_key: str, crew: Any, environment: str = "production") -> None:
        self._crew = crew
        self._agent = Agent(
            api_key=api_key,
            agent_name="crewai-crew",
            environment=environment,
            framework="crewai",
        )

    async def kickoff(self) -> Any:
        await self._agent.track(action="crew_kickoff", target="crew")
        result = self._crew.kickoff()
        await self._agent.track(action="crew_complete", target="crew")
        return result
