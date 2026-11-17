"""Local LangChain callback and review gate. No network and no model API key."""

import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool

from layr import Agent, CallbackReviewBackend, ReviewResolution
from layr.integrations.langchain import LayrAsyncCallbackHandler, require_review_tool


def _approve(request):
    return ReviewResolution(
        outcome="approved",
        reviewer=request.reviewer,
        reason="ok",
    )


@tool
def lookup(city: str) -> str:
    """Look up a city."""
    return f"notes for {city}"


async def main() -> None:
    agent = Agent(
        api_key="local",
        agent_name="langchain-demo",
        exporter="local",
        mode="local",
        review_backend=CallbackReviewBackend(_approve),
    )
    handler = LayrAsyncCallbackHandler(agent)
    model = FakeListChatModel(responses=["Hello from the fake model"])
    reply = await model.ainvoke("Say hello", config={"callbacks": [handler]})
    print(reply.content)
    gated = require_review_tool(
        lookup,
        agent,
        reviewer="ava@example.com",
        decision_id_fn=lambda tool_input: str(tool_input["city"]),
        timeout_s=30,
        on_timeout="deny",
    )
    print(await gated.ainvoke({"city": "Paris"}))
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
