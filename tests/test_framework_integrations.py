import asyncio
import subprocess
import sys
from typing import Any, List

import pytest

from layr.agent import Agent
from layr.exporters.base import BaseExporter
from layr.integrations.autogen import (
    ToolReviewDenied as AutoGenDenied,
    message_callback,
    require_review_tool as autogen_review_tool,
)
from layr.integrations.crewai import (
    ToolReviewDenied as CrewDenied,
    require_review_tool as crew_review_tool,
    step_callback,
    task_callback,
)
from layr.review import CallbackReviewBackend, ReviewResolution
from layr.schema import LayrEvent


class Capture(BaseExporter):
    def __init__(self) -> None:
        self.events: List[LayrEvent] = []

    async def export(self, events: Any) -> None:
        self.events.extend(events)


class Step:
    def __init__(self) -> None:
        self.tool = "search"
        self.tool_input = "secret query"
        self.log = "secret log"


class TaskOutput:
    def __init__(self) -> None:
        self.name = "draft"
        self.raw = "secret output"
        self.description = "write the note"


class CrewTool:
    def __init__(self) -> None:
        self.name = "search"
        self.description = "Search"
        self.calls = 0

    def _run(self, query: str) -> str:
        self.calls += 1
        return f"result:{query}"

    def run(self, query: str) -> str:
        return self._run(query)

    async def arun(self, query: str) -> str:
        self.calls += 1
        return f"async:{query}"


class Message:
    def __init__(self) -> None:
        self.source = "user"
        self.content = "secret message"


class AutoGenTool:
    def __init__(self) -> None:
        self.name = "lookup"
        self.description = "Look up"
        self.calls = 0

    async def run(self, args: Any, cancellation_token: Any = None) -> str:
        self.calls += 1
        return "ok"

    async def run_json(
        self, args: Any, cancellation_token: Any = None, call_id: Any = None
    ) -> str:
        return await self.run(args, cancellation_token)


def _agent(**kwargs: Any) -> tuple[Agent, Capture]:
    capture = Capture()
    agent = Agent(
        api_key="k", exporter="local", mode="local", agent_name="fw", **kwargs
    )
    agent._exporter = capture
    return agent, capture


def _approve(request: Any) -> ReviewResolution:
    return ReviewResolution(outcome="approved", reviewer=request.reviewer, reason="ok")


def _override(request: Any) -> ReviewResolution:
    return ReviewResolution(
        outcome="overridden",
        reviewer=request.reviewer,
        reason="no",
        to_action="skip",
    )


def test_import_layr_does_not_import_frameworks() -> None:
    code = (
        "import layr, sys; "
        "bad = [n for n in ('crewai', 'autogen_core', 'autogen_agentchat') if n in sys.modules]; "
        "raise SystemExit(0 if not bad else 1)"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_crewai_callbacks_omit_content_unless_opted_in() -> None:
    agent, capture = _agent()
    step_callback(agent)(Step())
    task_callback(agent)(TaskOutput())
    joined = " ".join(str(event.metadata) for event in capture.events)
    assert "secret query" not in joined
    assert "secret output" not in joined
    assert capture.events[0].action_type == "search"
    assert capture.events[0].metadata["event"] == "step"
    assert capture.events[1].action_type == "task"
    assert capture.events[1].action_target == "draft"

    step_callback(agent, capture_content=True)(Step())
    assert capture.events[-1].metadata["tool_input"] == "secret query"
    asyncio.run(agent.aclose())


def test_crewai_review_gate() -> None:
    agent, _capture = _agent(review_backend=CallbackReviewBackend(_approve))
    tool = CrewTool()
    gated = crew_review_tool(
        tool,
        agent,
        reviewer="ava@example.com",
        decision_id_fn=lambda tool_input: str(tool_input),
        on_timeout="deny",
    )
    assert gated.run("paris") == "result:paris"
    assert tool.calls == 1

    agent.review_backend = CallbackReviewBackend(_override)
    blocked = crew_review_tool(
        tool, agent, reviewer="ava@example.com", on_timeout="deny"
    )
    with pytest.raises(CrewDenied) as raised:
        blocked.run("berlin")
    assert raised.value.outcome == "overridden"
    assert tool.calls == 1
    asyncio.run(agent.aclose())


@pytest.mark.asyncio
async def test_crewai_review_gate_timeout() -> None:
    agent, _capture = _agent(review_backend=CallbackReviewBackend())
    tool = CrewTool()
    gated = crew_review_tool(
        tool,
        agent,
        reviewer="ava@example.com",
        timeout_s=0,
        on_timeout="deny",
    )
    with pytest.raises(CrewDenied) as raised:
        await gated.arun("rome")
    assert raised.value.outcome == "timed_out"
    assert tool.calls == 0
    await agent.aclose()


def test_autogen_messages_omit_content_unless_opted_in() -> None:
    agent, capture = _agent()
    message_callback(agent)(Message())
    assert capture.events[0].action_type == "message"
    assert capture.events[0].action_target == "user"
    assert "secret message" not in str(capture.events[0].metadata)
    message_callback(agent, capture_content=True)(Message())
    assert capture.events[-1].metadata["content"] == "secret message"
    asyncio.run(agent.aclose())


@pytest.mark.asyncio
async def test_autogen_review_gate_function_and_tool() -> None:
    agent, capture = _agent(review_backend=CallbackReviewBackend(_approve))
    calls = {"n": 0}

    def lookup(city: str) -> str:
        calls["n"] += 1
        return f"notes:{city}"

    gated = autogen_review_tool(
        lookup,
        agent,
        reviewer="ava@example.com",
        decision_id_fn=lambda tool_input: str(tool_input["city"]),
        on_timeout="deny",
    )
    assert await gated(city="Paris") == "notes:Paris"
    assert calls["n"] == 1
    assert "notes:Paris" not in str(capture.events[-1].metadata)

    tool = AutoGenTool()
    proxy = autogen_review_tool(
        tool, agent, reviewer="ava@example.com", on_timeout="deny"
    )
    assert await proxy.run({"city": "Lyon"}) == "ok"
    assert tool.calls == 1

    agent.review_backend = CallbackReviewBackend(_override)
    denied = autogen_review_tool(
        tool, agent, reviewer="ava@example.com", on_timeout="deny"
    )
    with pytest.raises(AutoGenDenied) as overridden:
        await denied.run_json({"city": "Berlin"})
    assert overridden.value.outcome == "overridden"
    assert tool.calls == 1

    agent.review_backend = CallbackReviewBackend()
    waiting = autogen_review_tool(
        lookup,
        agent,
        reviewer="ava@example.com",
        timeout_s=0,
        on_timeout="deny",
    )
    with pytest.raises(AutoGenDenied) as timed:
        await waiting(city="Rome")
    assert timed.value.outcome == "timed_out"
    assert calls["n"] == 1
    await agent.aclose()
