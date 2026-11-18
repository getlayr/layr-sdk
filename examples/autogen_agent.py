"""AutoGen 0.4-shaped message log and review gate. No model client is required.

``AssistantAgent`` accepts the async function returned by
``require_review_tool``. Legacy pyautogen 0.2 ``register_function`` is not
wrapped.
"""

import asyncio

from layr import Agent, CallbackReviewBackend, ReviewResolution
from layr.integrations.autogen import message_callback, require_review_tool


def _approve(request):
    return ReviewResolution(outcome="approved", reviewer=request.reviewer, reason="ok")


class TextMessage:
    def __init__(self, source: str, content: str) -> None:
        self.source = source
        self.content = content


def lookup(city: str) -> str:
    """Look up a city."""
    return f"notes for {city}"


async def main() -> None:
    agent = Agent(
        api_key="local",
        agent_name="autogen-demo",
        exporter="local",
        mode="local",
        review_backend=CallbackReviewBackend(_approve),
    )
    message_callback(agent)(TextMessage("user", "hidden prompt"))
    gated = require_review_tool(
        lookup,
        agent,
        reviewer="ava@example.com",
        decision_id_fn=lambda tool_input: str(tool_input["city"]),
        timeout_s=30,
        on_timeout="deny",
    )
    print(await gated(city="Paris"))
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
