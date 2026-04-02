import asyncio

from layr.integrations.crewai import LayrCrew


class MockCrew:
    def kickoff(self) -> str:
        return "done"


async def main() -> None:
    crew = LayrCrew(api_key="your-key", crew=MockCrew(), environment="production")
    result = await crew.kickoff()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
