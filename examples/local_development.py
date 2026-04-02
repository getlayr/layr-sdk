import asyncio

from layr import Agent


async def main() -> None:
    agent = Agent(api_key="your-key", exporter="local", mode="local", agent_name="dev-agent")
    await agent.track(action="send_email", target="user@example.com", input_tokens=450, output_tokens=210, latency_ms=1200)
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
