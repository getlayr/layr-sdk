"""Small LangChain callback demo. See examples/langchain_agent.py for the review gate."""

import asyncio

from layr import Agent
from layr.integrations.langchain import LayrAsyncCallbackHandler


async def main() -> None:
    agent = Agent(
        api_key="local",
        agent_name="customer-support-agent",
        exporter="local",
        mode="local",
    )
    handler = LayrAsyncCallbackHandler(agent)
    from uuid import uuid4

    run_id = uuid4()
    await handler.on_tool_start({"name": "search"}, "ticket 123", run_id=run_id)
    await handler.on_tool_end("found", run_id=run_id)
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
