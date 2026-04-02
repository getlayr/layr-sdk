import asyncio

from layr import Agent


async def main() -> None:
    agent = Agent(
        api_key="your-layer-api-key",
        agent_name="customer-support-agent",
        environment="production",
        base_url="http://localhost:3000",
    )
    await agent.track(
        action="send_email",
        target="customer@example.com",
        input_tokens=450,
        output_tokens=210,
        latency_ms=1200,
        model="gpt-4o",
        metadata={"subject": "Your order update"},
    )
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
