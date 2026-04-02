import asyncio

from layr import Agent


async def main() -> None:
    agent = Agent(
        api_key="your-key",
        exporter="otlp",
        mode="production",
        agent_name="otel-agent",
    )
    await agent.track(action="summarize", target="document", input_tokens=1000, output_tokens=200, latency_ms=980)
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
