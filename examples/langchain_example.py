import asyncio
from layr.integrations.langchain import LayrCallbackHandler


async def demo() -> None:
    cb = LayrCallbackHandler(api_key="your-key", agent_name="customer-support-agent")
    await cb.on_chain_start({"name": "support-chain"}, {"ticket_id": "123"})
    await cb.on_llm_end(response="ok", model="gpt-4o", input_tokens=120, output_tokens=80)
    await cb._agent.aclose()


if __name__ == "__main__":
    asyncio.run(demo())
