import asyncio
import subprocess
import sys
import uuid
from typing import Any, List

import pytest

from layr.agent import Agent
from layr.exporters.base import BaseExporter
from layr.integrations.langchain import token_counts
from layr.review import CallbackReviewBackend, ReviewResolution
from layr.schema import LayrEvent

langchain = pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
)  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, LLMResult  # noqa: E402
from langchain_core.tools import BaseTool  # noqa: E402

from layr.integrations.langchain import (  # noqa: E402
    LayrAsyncCallbackHandler,
    LayrCallbackHandler,
    ToolReviewDenied,
    require_review_tool,
)


class Capture(BaseExporter):
    def __init__(self) -> None:
        self.events: List[LayrEvent] = []

    async def export(self, events: Any) -> None:
        self.events.extend(events)


class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search a city"

    def _run(self, query: str) -> str:
        self._count = getattr(self, "_count", 0) + 1
        return f"result:{query}"

    async def _arun(self, query: str) -> str:
        self._count = getattr(self, "_count", 0) + 1
        return f"async:{query}"


def _agent(**kwargs: Any) -> tuple[Agent, Capture]:
    capture = Capture()
    agent = Agent(
        api_key="k",
        exporter="local",
        mode="local",
        agent_name="lc",
        **kwargs,
    )
    agent._exporter = capture
    return agent, capture


def _approve(request: Any) -> ReviewResolution:
    return ReviewResolution(
        outcome="approved",
        reviewer=request.reviewer,
        reason="ok",
    )


def test_core_import_does_not_import_langchain() -> None:
    code = (
        "import layr, sys; "
        "raise SystemExit(0 if 'langchain_core' not in sys.modules else 1)"
    )
    subprocess.check_call([sys.executable, "-c", code])


@pytest.mark.asyncio
async def test_async_handler_maps_tool_and_model_without_content() -> None:
    agent, capture = _agent()
    handler = LayrAsyncCallbackHandler(agent)
    model = FakeListChatModel(responses=["secret reply"])
    await model.ainvoke("secret prompt", config={"callbacks": [handler]})
    run_id = uuid.uuid4()
    await handler.on_tool_start(
        {"name": "search"},
        "secret query",
        run_id=run_id,
        inputs={"query": "secret query"},
    )
    await handler.on_tool_end("secret result", run_id=run_id)
    actions = [event.action_type for event in capture.events]
    assert "search" in actions
    joined = " ".join(str(event.metadata) for event in capture.events)
    assert "secret prompt" not in joined
    assert "secret reply" not in joined
    assert "secret query" not in joined
    assert "secret result" not in joined
    tool_end = [
        event
        for event in capture.events
        if event.metadata.get("langchain_event") == "tool_end"
    ]
    assert tool_end[-1].action_success is True
    assert tool_end[-1].action_target == str(run_id)
    await agent.aclose()


@pytest.mark.asyncio
async def test_token_capture_and_content_opt_in() -> None:
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="visible reply",
                        usage_metadata={
                            "input_tokens": 11,
                            "output_tokens": 4,
                            "total_tokens": 15,
                        },
                    )
                )
            ]
        ],
        llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )
    assert token_counts(response) == (11, 4)

    agent, capture = _agent()
    handler = LayrAsyncCallbackHandler(agent, capture_content=True)
    run_id = uuid.uuid4()
    await handler.on_chat_model_start(
        {"name": "chat", "kwargs": {"model": "fake-model"}},
        [[]],
        run_id=run_id,
    )
    await handler.on_llm_end(response, run_id=run_id)
    llm_end = capture.events[-1]
    assert llm_end.action_type == "fake-model"
    assert llm_end.input_tokens == 11
    assert llm_end.output_tokens == 4
    assert llm_end.metadata["response_type"] == "LLMResult"

    tool_id = uuid.uuid4()
    await handler.on_tool_start(
        {"name": "search"}, "paris", run_id=tool_id, inputs={"query": "paris"}
    )
    await handler.on_tool_end("sunny", run_id=tool_id)
    await handler.on_tool_error(RuntimeError("boom"), run_id=uuid.uuid4())
    assert capture.events[-3].metadata["input"] == "paris"
    assert capture.events[-2].metadata["output"] == "sunny"
    assert capture.events[-1].action_success is False
    assert capture.events[-1].metadata["error_type"] == "RuntimeError"
    assert capture.events[-1].metadata["error"] == "boom"
    await agent.aclose()


def test_sync_handler_records_tool_error() -> None:
    agent, capture = _agent()
    handler = LayrCallbackHandler(agent)
    run_id = uuid.uuid4()
    handler.on_tool_start({"name": "search"}, "hidden", run_id=run_id)
    handler.on_tool_error(TimeoutError("hidden"), run_id=run_id)
    assert capture.events[-1].action_type == "search"
    assert capture.events[-1].action_success is False
    assert "error" not in capture.events[-1].metadata
    asyncio.run(agent.aclose())


def test_require_review_tool_sync_approve_and_override() -> None:
    agent, _capture = _agent(review_backend=CallbackReviewBackend(_approve))
    tool = SearchTool()
    gated = require_review_tool(
        tool,
        agent,
        reviewer="ava@example.com",
        decision_id_fn=lambda tool_input: str(tool_input["query"]),
        timeout_s=5,
        on_timeout="deny",
    )
    assert gated.invoke({"query": "paris"}) == "result:paris"
    assert tool._count == 1

    def override(request: Any) -> ReviewResolution:
        return ReviewResolution(
            outcome="overridden",
            reviewer=request.reviewer,
            reason="do not search",
            to_action="skip",
        )

    agent.review_backend = CallbackReviewBackend(override)
    blocked = require_review_tool(
        tool, agent, reviewer="ava@example.com", on_timeout="deny"
    )
    with pytest.raises(ToolReviewDenied) as raised:
        blocked.invoke({"query": "berlin"})
    assert raised.value.outcome == "overridden"
    assert tool._count == 1
    asyncio.run(agent.aclose())


@pytest.mark.asyncio
async def test_require_review_tool_async_approve_and_timeout() -> None:
    agent, _capture = _agent(review_backend=CallbackReviewBackend(_approve))
    tool = SearchTool()
    gated = require_review_tool(
        tool,
        agent,
        reviewer="ava@example.com",
        decision_id_fn=lambda tool_input: str(tool_input["query"]),
        timeout_s=5,
        on_timeout="deny",
    )
    assert await gated.ainvoke({"query": "lyon"}) == "async:lyon"
    assert tool._count == 1

    agent.review_backend = CallbackReviewBackend()
    waiting = require_review_tool(
        tool,
        agent,
        reviewer="ava@example.com",
        timeout_s=0,
        on_timeout="deny",
    )
    with pytest.raises(ToolReviewDenied) as timed:
        await waiting.ainvoke({"query": "rome"})
    assert timed.value.outcome == "timed_out"
    assert tool._count == 1
    await agent.aclose()
