"""Optional LangChain callback and tool review gate.

Importing this module does not import LangChain. The classes are built on
first use and raise ImportError with install instructions when
``langchain-core`` is missing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, Mapping, Optional, Sequence, Tuple

from layr.agent import Agent
from layr.exceptions import ReviewError

_MISSING = (
    "LangChain support requires the optional extra. "
    "Install it with: pip install 'layr-sdk[langchain]'"
)

_BUILT: Dict[str, Any] = {}


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


def _reset_for_tests() -> None:
    _BUILT.clear()


def _import_langchain() -> Tuple[Any, Any, Any]:
    try:
        from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
        from langchain_core.tools import BaseTool
    except ImportError as exc:
        raise ImportError(_MISSING) from exc
    return BaseCallbackHandler, AsyncCallbackHandler, BaseTool


def _as_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def tool_name(serialized: Mapping[str, object]) -> str:
    name = serialized.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "tool"


def model_name(serialized: Mapping[str, object], extra: Mapping[str, object]) -> str:
    sources = (
        extra.get("invocation_params"),
        serialized.get("kwargs"),
        extra,
    )
    for source in sources:
        data = _mapping(source)
        for key in ("model", "model_name"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    name = serialized.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    ident = serialized.get("id")
    if isinstance(ident, Sequence) and not isinstance(ident, (str, bytes)):
        for item in reversed(list(ident)):
            if isinstance(item, str) and item.strip():
                return item.strip()
    return "llm"


def token_counts(response: object) -> Tuple[int, int]:
    """Read token counts from llm_output or message usage_metadata."""
    input_tokens = 0
    output_tokens = 0
    llm_output = _mapping(getattr(response, "llm_output", None))
    usage = _mapping(llm_output.get("token_usage") or llm_output.get("usage"))
    if not usage:
        usage = llm_output
    input_tokens = _as_int(
        usage.get("prompt_tokens", usage.get("input_tokens", input_tokens))
    )
    output_tokens = _as_int(
        usage.get("completion_tokens", usage.get("output_tokens", output_tokens))
    )
    generations = getattr(response, "generations", None)
    if isinstance(generations, Sequence) and not isinstance(generations, (str, bytes)):
        for group in generations:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                continue
            for generation in group:
                message = getattr(generation, "message", generation)
                metadata = _mapping(getattr(message, "usage_metadata", None))
                if not metadata:
                    continue
                input_tokens = _as_int(metadata.get("input_tokens", input_tokens))
                output_tokens = _as_int(metadata.get("output_tokens", output_tokens))
    return input_tokens, output_tokens


def _message_texts(messages: object) -> list[str]:
    texts: list[str] = []
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return texts
    for batch in messages:
        if not isinstance(batch, Sequence) or isinstance(batch, (str, bytes)):
            continue
        for message in batch:
            content = getattr(message, "content", "")
            texts.append(content if isinstance(content, str) else str(content))
    return texts


class _RunLog:
    def __init__(self, agent: Agent, capture_content: bool) -> None:
        self.agent = agent
        self.capture_content = capture_content
        self._started: Dict[str, float] = {}
        self._models: Dict[str, str] = {}

    def note_start(self, run_id: object, model: str = "") -> None:
        self._started.setdefault(str(run_id), time.perf_counter())
        if model:
            self._models.setdefault(str(run_id), model)

    def latency_ms(self, run_id: object) -> int:
        started = self._started.pop(str(run_id), None)
        if started is None:
            return 0
        return max(0, int((time.perf_counter() - started) * 1000))

    async def on_tool_start(
        self,
        serialized: Mapping[str, object],
        input_str: str,
        run_id: object,
        inputs: object,
    ) -> None:
        self.note_start(run_id)
        metadata: Dict[str, object] = {
            "run_id": str(run_id),
            "langchain_event": "tool_start",
        }
        if self.capture_content:
            metadata["input"] = input_str
            if inputs is not None:
                metadata["inputs"] = inputs
        await self.agent.track(
            action=tool_name(serialized),
            target=str(run_id),
            success=True,
            metadata=metadata,
        )

    async def on_tool_end(self, output: object, run_id: object, name: str) -> None:
        metadata: Dict[str, object] = {
            "run_id": str(run_id),
            "langchain_event": "tool_end",
        }
        if self.capture_content:
            metadata["output"] = output if isinstance(output, str) else str(output)
        await self.agent.track(
            action=name,
            target=str(run_id),
            success=True,
            latency_ms=self.latency_ms(run_id),
            metadata=metadata,
        )

    async def on_tool_error(
        self, error: BaseException, run_id: object, name: str
    ) -> None:
        metadata: Dict[str, object] = {
            "run_id": str(run_id),
            "langchain_event": "tool_error",
            "error_type": type(error).__name__,
        }
        if self.capture_content:
            metadata["error"] = str(error)
        await self.agent.track(
            action=name,
            target=str(run_id),
            success=False,
            latency_ms=self.latency_ms(run_id),
            metadata=metadata,
        )

    async def on_chat_model_start(
        self,
        serialized: Mapping[str, object],
        messages: object,
        run_id: object,
        extra: Mapping[str, object],
    ) -> None:
        model = model_name(serialized, extra)
        self.note_start(run_id, model)
        texts = _message_texts(messages)
        metadata: Dict[str, object] = {
            "run_id": str(run_id),
            "langchain_event": "chat_model_start",
            "message_count": len(texts),
        }
        if self.capture_content:
            metadata["messages"] = texts
        await self.agent.track(
            action=model,
            target=str(run_id),
            success=True,
            metadata=metadata,
        )

    async def on_llm_end(
        self,
        response: object,
        run_id: object,
        extra: Mapping[str, object],
    ) -> None:
        model = self._models.pop(str(run_id), "") or model_name({}, extra)
        input_tokens, output_tokens = token_counts(response)
        metadata: Dict[str, object] = {
            "run_id": str(run_id),
            "langchain_event": "llm_end",
        }
        if self.capture_content:
            metadata["response_type"] = type(response).__name__
        await self.agent.track(
            action=model,
            target=str(run_id),
            success=True,
            latency_ms=self.latency_ms(run_id),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata,
        )


def _run_sync(coro: Coroutine[Any, Any, None]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    asyncio.get_running_loop().create_task(coro)


def _tool_input(args: Tuple[object, ...], kwargs: Mapping[str, object]) -> object:
    if kwargs and not args:
        return dict(kwargs)
    if len(args) == 1 and not kwargs:
        return args[0]
    if args and kwargs:
        combined = dict(kwargs)
        combined["args"] = args
        return combined
    return {}


def _decision_id(
    tool_name_value: str,
    tool_input: object,
    decision_id_fn: Optional[Callable[[object], str]],
) -> str:
    if decision_id_fn is None:
        return tool_name_value
    return decision_id_fn(tool_input)


def _reject_unless_approved(name: str, outcome: str, reason: str) -> None:
    if outcome == "approved":
        return
    raise ToolReviewDenied(name, outcome, reason)


def _build() -> Tuple[type, type, Callable[..., Any]]:
    if "handler" in _BUILT:
        return _BUILT["handler"], _BUILT["async_handler"], _BUILT["require_review_tool"]

    base_handler, async_handler, base_tool = _import_langchain()

    class LayrCallbackHandler(base_handler):  # type: ignore[misc, valid-type]
        """Sync LangChain callback that records tool and model calls.

        Prompt and response text is omitted unless ``capture_content`` is true.
        """

        def __init__(self, agent: Agent, *, capture_content: bool = False) -> None:
            super().__init__()
            self._agent = agent
            self._capture_content = capture_content
            self._log = _RunLog(agent, capture_content)
            self._tool_names: Dict[str, str] = {}

        def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            inputs: Any = None,
            **_: Any,
        ) -> None:
            self._tool_names[str(run_id)] = tool_name(serialized)
            _run_sync(self._log.on_tool_start(serialized, input_str, run_id, inputs))

        def on_tool_end(self, output: Any, *, run_id: Any, **_: Any) -> None:
            name = self._tool_names.pop(str(run_id), "tool")
            _run_sync(self._log.on_tool_end(output, run_id, name))

        def on_tool_error(self, error: BaseException, *, run_id: Any, **_: Any) -> None:
            name = self._tool_names.pop(str(run_id), "tool")
            _run_sync(self._log.on_tool_error(error, run_id, name))

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: list,
            *,
            run_id: Any,
            **kwargs: Any,
        ) -> None:
            self._log.note_start(run_id, model_name(serialized, kwargs))

        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: list,
            *,
            run_id: Any,
            **kwargs: Any,
        ) -> None:
            _run_sync(
                self._log.on_chat_model_start(serialized, messages, run_id, kwargs)
            )

        def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
            _run_sync(self._log.on_llm_end(response, run_id, kwargs))

    class LayrAsyncCallbackHandler(async_handler):  # type: ignore[misc, valid-type]
        """Async LangChain callback that records tool and model calls."""

        def __init__(self, agent: Agent, *, capture_content: bool = False) -> None:
            super().__init__()
            self._agent = agent
            self._capture_content = capture_content
            self._log = _RunLog(agent, capture_content)
            self._tool_names: Dict[str, str] = {}

        async def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            *,
            run_id: Any,
            inputs: Any = None,
            **_: Any,
        ) -> None:
            self._tool_names[str(run_id)] = tool_name(serialized)
            await self._log.on_tool_start(serialized, input_str, run_id, inputs)

        async def on_tool_end(self, output: Any, *, run_id: Any, **_: Any) -> None:
            name = self._tool_names.pop(str(run_id), "tool")
            await self._log.on_tool_end(output, run_id, name)

        async def on_tool_error(
            self, error: BaseException, *, run_id: Any, **_: Any
        ) -> None:
            name = self._tool_names.pop(str(run_id), "tool")
            await self._log.on_tool_error(error, run_id, name)

        async def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: list,
            *,
            run_id: Any,
            **kwargs: Any,
        ) -> None:
            self._log.note_start(run_id, model_name(serialized, kwargs))

        async def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: list,
            *,
            run_id: Any,
            **kwargs: Any,
        ) -> Any:
            await self._log.on_chat_model_start(serialized, messages, run_id, kwargs)

        async def on_llm_end(
            self, response: Any, *, run_id: Any, **kwargs: Any
        ) -> None:
            await self._log.on_llm_end(response, run_id, kwargs)

    def require_review_tool(
        tool: Any,
        agent: Agent,
        *,
        reviewer: str,
        decision_id_fn: Optional[Callable[[object], str]] = None,
        timeout_s: float = 3600,
        on_timeout: str = "deny",
    ) -> Any:
        """Return a tool that runs only after ``agent.require_review`` is approved.

        Any other outcome, including override and timeout, raises
        ``ToolReviewDenied`` and does not call the original tool.
        ``on_timeout`` defaults to ``deny``.
        """
        if not isinstance(tool, base_tool):
            raise TypeError("tool must be a LangChain BaseTool")

        class ReviewedTool(base_tool):  # type: ignore[misc, valid-type]
            name: str = tool.name
            description: str = tool.description or ""
            args_schema: Any = tool.args_schema
            return_direct: bool = bool(getattr(tool, "return_direct", False))

            def _run(self, *args: Any, **kwargs: Any) -> Any:
                tool_input = _tool_input(args, kwargs)
                decision = agent.require_review_sync(
                    decision_id=_decision_id(tool.name, tool_input, decision_id_fn),
                    action=tool.name,
                    summary={"tool": tool.name, "input": tool_input},
                    reviewer=reviewer,
                    timeout_s=timeout_s,
                    on_timeout=on_timeout,
                )
                _reject_unless_approved(tool.name, decision.outcome, decision.reason)
                return tool.invoke(tool_input)

            async def _arun(self, *args: Any, **kwargs: Any) -> Any:
                tool_input = _tool_input(args, kwargs)
                decision = await agent.require_review(
                    decision_id=_decision_id(tool.name, tool_input, decision_id_fn),
                    action=tool.name,
                    summary={"tool": tool.name, "input": tool_input},
                    reviewer=reviewer,
                    timeout_s=timeout_s,
                    on_timeout=on_timeout,
                )
                _reject_unless_approved(tool.name, decision.outcome, decision.reason)
                return await tool.ainvoke(tool_input)

        return ReviewedTool()

    _BUILT["handler"] = LayrCallbackHandler
    _BUILT["async_handler"] = LayrAsyncCallbackHandler
    _BUILT["require_review_tool"] = require_review_tool
    return LayrCallbackHandler, LayrAsyncCallbackHandler, require_review_tool


def __getattr__(name: str) -> Any:
    if name in {
        "LayrCallbackHandler",
        "LayrAsyncCallbackHandler",
        "require_review_tool",
    }:
        handler, async_handler, review_tool = _build()
        values = {
            "LayrCallbackHandler": handler,
            "LayrAsyncCallbackHandler": async_handler,
            "require_review_tool": review_tool,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
