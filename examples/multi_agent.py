import asyncio

from layr import AgentNetwork


async def main() -> None:
    network = AgentNetwork(api_key="your-key")
    orchestrator = network.agent("orchestrator")
    researcher = network.agent("researcher")
    writer = network.agent("writer")

    await network.track_handoff(
        from_agent=orchestrator,
        to_agent=researcher,
        task="find Q3 sales data",
        context={"source": "warehouse"},
    )
    await network.track_handoff(
        from_agent=researcher,
        to_agent=writer,
        task="draft summary",
        context={"audience": "exec"},
    )
    await orchestrator.aclose()
    await researcher.aclose()
    await writer.aclose()


if __name__ == "__main__":
    asyncio.run(main())
