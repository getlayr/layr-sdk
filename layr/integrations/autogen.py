from __future__ import annotations

from typing import Any

from layr.agent import Agent


class LayrConversableAgent:
    def __init__(
        self,
        api_key: str,
        name: str,
        environment: str = "production",
        llm_config: Any = None,
    ) -> None:
        self._name = name
        self._llm_config = llm_config
        self._agent = Agent(
            api_key=api_key,
            agent_name=name,
            environment=environment,
            framework="autogen",
        )

    async def send(self, message: str) -> None:
        await self._agent.track(action="autogen_send", target=self._name, metadata={"message": message[:500]})

    async def receive(self, message: str) -> None:
        await self._agent.track(action="autogen_receive", target=self._name, metadata={"message": message[:500]})
