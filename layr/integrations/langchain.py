from __future__ import annotations

import time
from typing import Dict, List, Optional

from layr.agent import Agent


class LayrCallbackHandler:
    def __init__(self, api_key: str, agent_name: str, environment: str = "production") -> None:
        self._agent = Agent(
            api_key=api_key,
            agent_name=agent_name,
            environment=environment,
            framework="langchain",
        )
        self._started_at: float = 0.0

    async def on_chain_start(self, serialized: Dict[str, str], inputs: Dict[str, str], **_: object) -> None:
        self._started_at = time.time()
        await self._agent.track(action="chain_start", target=serialized.get("name", "chain"), metadata=inputs)

    async def on_llm_end(
        self,
        response: object,
        *,
        model: str = "gpt-4o",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        latency_ms = int((time.time() - self._started_at) * 1000)
        await self._agent.track(
            action="llm_call",
            target=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            model=model,
            metadata={"response_type": str(type(response))},
        )

    async def on_tool_end(self, output: str, *, tool_name: str = "tool", latency_ms: int = 0) -> None:
        await self._agent.track(
            action="tool_use",
            target=tool_name,
            tools_used=[tool_name],
            latency_ms=latency_ms,
            metadata={"output": output[:500]},
        )

    async def on_chain_error(self, error: BaseException, **_: object) -> None:
        await self._agent.track(
            action="chain_error",
            target="chain",
            success=False,
            metadata={"error": str(error)},
        )
