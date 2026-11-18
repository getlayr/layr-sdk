"""Optional CrewAI callbacks and a tool review gate.

Importing this module does not import CrewAI. ``require_review_tool`` imports
CrewAI only when the object is a ``crewai.tools.BaseTool``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, Optional

from layr.agent import Agent
from layr.exceptions import ReviewError


class ToolReviewDenied(ReviewError):
    """The tool was not run because the review was not approved."""

    def __init__(self, tool_name: str, outcome: str, reason: str) -> None:
        self.tool_name = tool_name
        self.outcome = outcome
        self.reason = reason
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"tool {tool_name} was not run; review outcome was {outcome}{detail}"
        )


class LayrCrew:
    """Record crew kickoff and completion without importing CrewAI."""

    def __init__(
        self,
        api_key: str,
        crew: Any,
        environment: str = "production",
        *,
        capture_content: bool = False,
    ) -> None:
        self._crew = crew
        self._capture_content = capture_content
        self._agent = Agent(
            api_key=api_key,
            agent_name="crewai-crew",
            environment=environment,
            framework="crewai",
        )

    def step_callback(self) -> Callable[[object], None]:
        return step_callback(self._agent, capture_content=self._capture_content)

    def task_callback(self) -> Callable[[object], None]:
        return task_callback(self._agent, capture_content=self._capture_content)

    async def kickoff(self) -> Any:
        await self._agent.track(action="crew_kickoff", target="crew", success=True)
        result = self._crew.kickoff()
        if asyncio.iscoroutine(result):
            result = await result
        metadata: Dict[str, object] = {"framework": "crewai", "event": "kickoff"}
        if self._capture_content and result is not None:
            metadata["output"] = result if isinstance(result, str) else str(result)
        await self._agent.track(
            action="crew_complete",
            target="crew",
            success=True,
            metadata=metadata,
        )
        return result


def step_callback(
    agent: Agent, *, capture_content: bool = False
) -> Callable[[object], None]:
    """Return a CrewAI ``step_callback``.

    The callback records the step type and tool name. Tool input and logs are
    included only when ``capture_content`` is true. CrewAI accepts a sync
    callback and will schedule it if it returns a coroutine.
    """

    def _callback(step: object) -> None:
        _run_sync(_track_step(agent, step, capture_content))

    return _callback


def task_callback(
    agent: Agent, *, capture_content: bool = False
) -> Callable[[object], None]:
    """Return a CrewAI ``task_callback`` or per-task ``callback``."""

    def _callback(output: object) -> None:
        _run_sync(_track_task(agent, output, capture_content))

    return _callback


def instrument_crew(crew: Any, agent: Agent, *, capture_content: bool = False) -> Any:
    """Set empty ``step_callback`` and ``task_callback`` on a crew-like object."""
    if getattr(crew, "step_callback", None) is None:
        crew.step_callback = step_callback(agent, capture_content=capture_content)
    if getattr(crew, "task_callback", None) is None:
        crew.task_callback = task_callback(agent, capture_content=capture_content)
    return crew


def require_review_tool(
    tool: Any,
    agent: Agent,
    *,
    reviewer: str,
    decision_id_fn: Optional[Callable[[object], str]] = None,
    timeout_s: float = 3600,
    on_timeout: str = "deny",
    capture_content: bool = False,
) -> Any:
    """Return a tool that runs only after ``agent.require_review`` is approved.

    Sync ``run`` / ``_run`` and async ``arun`` / ``_arun`` are both gated.
    Any outcome other than ``approved`` raises ``ToolReviewDenied``.
    ``on_timeout`` defaults to ``deny``.

    Pass the wrapped tool to the CrewAI agent. CrewAI copies ``_run`` when it
    builds its structured tool, so wrapping after that copy does not gate the call.
    """
    if not hasattr(tool, "_run") and not callable(getattr(tool, "run", None)):
        raise TypeError("tool must provide _run or run")

    name = str(getattr(tool, "name", "tool"))
    description = str(getattr(tool, "description", "") or "")

    def _input(args: tuple, kwargs: dict) -> object:
        if kwargs and not args:
            return dict(kwargs)
        if len(args) == 1 and not kwargs:
            return args[0]
        if args or kwargs:
            return {"args": args, "kwargs": kwargs}
        return {}

    def _decision_id(tool_input: object) -> str:
        if decision_id_fn is None:
            return name
        return decision_id_fn(tool_input)

    def _gate_sync(tool_input: object) -> None:
        decision = agent.require_review_sync(
            decision_id=_decision_id(tool_input),
            action=name,
            summary={"tool": name, "input": tool_input},
            reviewer=reviewer,
            timeout_s=timeout_s,
            on_timeout=on_timeout,
        )
        if decision.outcome != "approved":
            raise ToolReviewDenied(name, decision.outcome, decision.reason)

    async def _gate_async(tool_input: object) -> None:
        decision = await agent.require_review(
            decision_id=_decision_id(tool_input),
            action=name,
            summary={"tool": name, "input": tool_input},
            reviewer=reviewer,
            timeout_s=timeout_s,
            on_timeout=on_timeout,
        )
        if decision.outcome != "approved":
            raise ToolReviewDenied(name, decision.outcome, decision.reason)

    async def _track_result(tool_input: object, result: object, success: bool) -> None:
        metadata: Dict[str, object] = {"framework": "crewai", "event": "tool"}
        if capture_content:
            metadata["input"] = tool_input
            metadata["output"] = result if isinstance(result, str) else str(result)
        await agent.track(
            action=name,
            target=_decision_id(tool_input),
            success=success,
            metadata=metadata,
        )

    class ReviewedTool:
        def __init__(self) -> None:
            self.name = name
            self.description = description
            self._tool = tool

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            tool_input = _input(args, kwargs)
            _gate_sync(tool_input)
            result = self._tool._run(*args, **kwargs)
            _run_sync(_track_result(tool_input, result, True))
            return result

        def run(self, *args: Any, **kwargs: Any) -> Any:
            if (
                hasattr(self._tool, "run")
                and type(self._tool).run is not ReviewedTool.run
            ):
                tool_input = _input(args, kwargs)
                _gate_sync(tool_input)
                result = self._tool.run(*args, **kwargs)
                _run_sync(_track_result(tool_input, result, True))
                return result
            return self._run(*args, **kwargs)

        async def _arun(self, *args: Any, **kwargs: Any) -> Any:
            tool_input = _input(args, kwargs)
            await _gate_async(tool_input)
            arun = getattr(self._tool, "_arun", None)
            if arun is None:
                result = self._tool._run(*args, **kwargs)
            else:
                result = arun(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
            await _track_result(tool_input, result, True)
            return result

        async def arun(self, *args: Any, **kwargs: Any) -> Any:
            if hasattr(self._tool, "arun"):
                tool_input = _input(args, kwargs)
                await _gate_async(tool_input)
                result = self._tool.arun(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                await _track_result(tool_input, result, True)
                return result
            return await self._arun(*args, **kwargs)

    return ReviewedTool()


def _run_sync(coro: Coroutine[Any, Any, None]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    asyncio.get_running_loop().create_task(coro)


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value)


async def _track_step(agent: Agent, step: object, capture_content: bool) -> None:
    tool = getattr(step, "tool", None)
    action = (
        tool.strip() if isinstance(tool, str) and tool.strip() else type(step).__name__
    )
    metadata: Dict[str, object] = {
        "framework": "crewai",
        "event": "step",
        "step_type": type(step).__name__,
    }
    if capture_content:
        for attr in ("tool_input", "log", "output", "text"):
            if hasattr(step, attr):
                metadata[attr] = _text(getattr(step, attr))
    await agent.track(
        action=action,
        target=type(step).__name__,
        success=True,
        metadata=metadata,
    )


async def _track_task(agent: Agent, output: object, capture_content: bool) -> None:
    name = (
        getattr(output, "name", None) or getattr(output, "description", None) or "task"
    )
    metadata: Dict[str, object] = {"framework": "crewai", "event": "task"}
    if capture_content:
        for attr in ("raw", "description", "output"):
            if hasattr(output, attr):
                metadata[attr] = _text(getattr(output, attr))
    await agent.track(
        action="task",
        target=_text(name),
        success=True,
        metadata=metadata,
    )
