"""Optional AutoGen 0.4+ message logging and a tool review gate.

The maintained packages are ``autogen-agentchat`` and ``autogen-core``
(0.4 and later). Legacy ``pyautogen`` 0.2 is not the target.

Importing this module does not import AutoGen. ``require_review_tool``
accepts a callable or any object with ``name`` and an async ``run``.
"""

from __future__ import annotations

import asyncio
import inspect
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


class LayrConversableAgent:
    """Record sent and received messages without importing AutoGen.

    Message text is omitted unless ``capture_content`` is true.
    """

    def __init__(
        self,
        api_key: str,
        name: str,
        environment: str = "production",
        llm_config: Any = None,
        *,
        capture_content: bool = False,
    ) -> None:
        self._name = name
        self._llm_config = llm_config
        self._capture_content = capture_content
        self._agent = Agent(
            api_key=api_key,
            agent_name=name,
            environment=environment,
            framework="autogen",
        )

    async def send(self, message: str) -> None:
        await _track_message(
            self._agent, _Text(self._name, message), "send", self._capture_content
        )

    async def receive(self, message: str) -> None:
        await _track_message(
            self._agent, _Text(self._name, message), "receive", self._capture_content
        )


class _Text:
    def __init__(self, source: str, content: str) -> None:
        self.source = source
        self.content = content

    def __class_name__(self) -> str:
        return "TextMessage"


def message_callback(
    agent: Agent, *, capture_content: bool = False
) -> Callable[[object], None]:
    """Return a callback for an AutoGen message or event.

    Records the message type and source. Content is included only when
    ``capture_content`` is true. AutoGen 0.4 does not have one global
    human-input hook; pass tools through ``require_review_tool`` for approval.
    """

    def _callback(message: object) -> None:
        _run_sync(_track_message(agent, message, "message", capture_content))

    return _callback


def require_review_tool(
    tool: Any,
    agent: Agent,
    *,
    reviewer: str,
    decision_id_fn: Optional[Callable[[object], str]] = None,
    timeout_s: float = 3600,
    on_timeout: str = "deny",
    capture_content: bool = False,
    description: str = "",
    name: Optional[str] = None,
) -> Any:
    """Gate a function or AutoGen tool on ``agent.require_review``.

    A function is returned as an async wrapper that ``AssistantAgent`` can
    pass in ``tools=[...]``. AutoGen wraps that function in ``FunctionTool``.
    An object with ``name`` and async ``run`` is returned as a proxy whose
    ``run`` and ``run_json`` review first.

    The tool runs only when the outcome is ``approved``. Otherwise
    ``ToolReviewDenied`` is raised. ``on_timeout`` defaults to ``deny``.

    Legacy pyautogen 0.2 ``register_function`` is not wrapped. Use an
    AutoGen 0.4 tool or a plain function.
    """
    if _is_tool_object(tool):
        return _proxy_tool(
            tool,
            agent,
            reviewer=reviewer,
            decision_id_fn=decision_id_fn,
            timeout_s=timeout_s,
            on_timeout=on_timeout,
            capture_content=capture_content,
        )
    if not callable(tool):
        raise TypeError("tool must be a callable or an object with name and run")
    return _wrap_function(
        tool,
        agent,
        reviewer=reviewer,
        decision_id_fn=decision_id_fn,
        timeout_s=timeout_s,
        on_timeout=on_timeout,
        capture_content=capture_content,
        description=description,
        name=name,
    )


def _is_tool_object(tool: Any) -> bool:
    return (
        hasattr(tool, "run") and hasattr(tool, "name") and not inspect.isfunction(tool)
    )


def _tool_name(tool: Any, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "tool"


def _decision_id(
    fallback: str, tool_input: object, decision_id_fn: Optional[Callable[[object], str]]
) -> str:
    if decision_id_fn is None:
        return fallback
    return decision_id_fn(tool_input)


def _input_payload(value: object) -> object:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return value


async def _review(
    agent: Agent,
    *,
    tool_name: str,
    tool_input: object,
    reviewer: str,
    decision_id_fn: Optional[Callable[[object], str]],
    timeout_s: float,
    on_timeout: str,
) -> None:
    decision = await agent.require_review(
        decision_id=_decision_id(tool_name, tool_input, decision_id_fn),
        action=tool_name,
        summary={"tool": tool_name, "input": tool_input},
        reviewer=reviewer,
        timeout_s=timeout_s,
        on_timeout=on_timeout,
    )
    if decision.outcome != "approved":
        raise ToolReviewDenied(tool_name, decision.outcome, decision.reason)


async def _track_tool(
    agent: Agent,
    tool_name: str,
    tool_input: object,
    result: object,
    capture_content: bool,
) -> None:
    metadata: Dict[str, object] = {"framework": "autogen", "event": "tool"}
    if capture_content:
        metadata["input"] = tool_input
        metadata["output"] = result if isinstance(result, str) else str(result)
    await agent.track(
        action=tool_name,
        target=_decision_id(tool_name, tool_input, None),
        success=True,
        metadata=metadata,
    )


def _wrap_function(
    func: Callable[..., Any],
    agent: Agent,
    *,
    reviewer: str,
    decision_id_fn: Optional[Callable[[object], str]],
    timeout_s: float,
    on_timeout: str,
    capture_content: bool,
    description: str,
    name: Optional[str],
) -> Callable[..., Any]:
    tool_name = _tool_name(func, name)

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs and not args:
            tool_input: object = dict(kwargs)
        elif len(args) == 1 and not kwargs:
            tool_input = args[0]
        else:
            tool_input = {"args": args, "kwargs": kwargs}
        await _review(
            agent,
            tool_name=tool_name,
            tool_input=tool_input,
            reviewer=reviewer,
            decision_id_fn=decision_id_fn,
            timeout_s=timeout_s,
            on_timeout=on_timeout,
        )
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        await _track_tool(agent, tool_name, tool_input, result, capture_content)
        return result

    wrapped.__name__ = tool_name
    wrapped.__doc__ = description or (func.__doc__ or "")
    wrapped.__annotations__ = dict(getattr(func, "__annotations__", {}))
    wrapped.__wrapped__ = func  # type: ignore[attr-defined]
    return wrapped


def _proxy_tool(
    tool: Any,
    agent: Agent,
    *,
    reviewer: str,
    decision_id_fn: Optional[Callable[[object], str]],
    timeout_s: float,
    on_timeout: str,
    capture_content: bool,
) -> Any:
    tool_name = _tool_name(tool, None)

    class ReviewedTool:
        def __init__(self) -> None:
            self._tool = tool
            self.name = tool_name
            self.description = str(getattr(tool, "description", "") or "")

        def __getattr__(self, item: str) -> Any:
            if item in {"run", "run_json"}:
                raise AttributeError(item)
            return getattr(self._tool, item)

        async def run(
            self, args: Any, cancellation_token: Any = None, *rest: Any
        ) -> Any:
            tool_input = _input_payload(args)
            await _review(
                agent,
                tool_name=self.name,
                tool_input=tool_input,
                reviewer=reviewer,
                decision_id_fn=decision_id_fn,
                timeout_s=timeout_s,
                on_timeout=on_timeout,
            )
            result = await self._tool.run(args, cancellation_token, *rest)
            await _track_tool(agent, self.name, tool_input, result, capture_content)
            return result

        async def run_json(
            self,
            args: Any,
            cancellation_token: Any = None,
            call_id: Any = None,
        ) -> Any:
            tool_input = _input_payload(args)
            await _review(
                agent,
                tool_name=self.name,
                tool_input=tool_input,
                reviewer=reviewer,
                decision_id_fn=decision_id_fn,
                timeout_s=timeout_s,
                on_timeout=on_timeout,
            )
            result = await self._tool.run_json(args, cancellation_token, call_id)
            await _track_tool(agent, self.name, tool_input, result, capture_content)
            return result

    return ReviewedTool()


async def _track_message(
    agent: Agent, message: object, event: str, capture_content: bool
) -> None:
    source = getattr(message, "source", None)
    target = source if isinstance(source, str) and source else type(message).__name__
    metadata: Dict[str, object] = {
        "framework": "autogen",
        "event": event,
        "message_type": type(message).__name__,
    }
    if isinstance(source, str):
        metadata["source"] = source
    if capture_content and hasattr(message, "content"):
        content = getattr(message, "content")
        metadata["content"] = content if isinstance(content, str) else str(content)
    action = "message" if event == "message" else f"autogen_{event}"
    await agent.track(action=action, target=target, success=True, metadata=metadata)


def _run_sync(coro: Coroutine[Any, Any, None]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    asyncio.get_running_loop().create_task(coro)
