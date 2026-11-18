"""CrewAI-shaped callbacks and a review gate. No CrewAI install is required.

Pass ``step_callback(agent)`` and ``task_callback(agent)`` to a real CrewAI
Agent or Crew. Give the agent the object returned by ``require_review_tool``
before CrewAI copies the tool's ``_run``.
"""

import asyncio

from layr import Agent, CallbackReviewBackend, ReviewResolution
from layr.integrations.crewai import require_review_tool, step_callback, task_callback


def _approve(request):
    return ReviewResolution(outcome="approved", reviewer=request.reviewer, reason="ok")


class Search:
    def __init__(self) -> None:
        self.name = "search"
        self.description = "Search a city"

    def _run(self, query: str) -> str:
        return f"notes for {query}"

    def run(self, query: str) -> str:
        return self._run(query)

    async def arun(self, query: str) -> str:
        return self._run(query)


class Step:
    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.tool_input = "hidden"


async def main() -> None:
    agent = Agent(
        api_key="local",
        agent_name="crewai-demo",
        exporter="local",
        mode="local",
        review_backend=CallbackReviewBackend(_approve),
    )
    step_callback(agent)(Step("search"))
    task_callback(agent)(type("TaskOutput", (), {"name": "draft", "raw": "hidden"})())
    gated = require_review_tool(
        Search(),
        agent,
        reviewer="ava@example.com",
        decision_id_fn=lambda tool_input: str(tool_input),
        timeout_s=30,
        on_timeout="deny",
    )
    print(await gated.arun("Paris"))
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
