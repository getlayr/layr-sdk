"""Human-oversight checkpoint and pluggable review backends.

``require_review`` pauses a high-risk decision until a named person approves,
overrides, or escalates. Two backends ship with the SDK:

- ``CallbackReviewBackend`` keeps the pending review in memory and can resolve
  it from a callback or from ``resolve()`` (another task, thread, or test).
- ``WebhookReviewBackend`` delivers the same request to an HTTP endpoint, or
  to a caller-supplied hook, and completes it when ``handle_callback`` is
  invoked with the signed body.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Union,
)

import httpx

from layr.evidence import canonical_json
from layr.exceptions import ReviewError
from layr.schema import LayrEvent

APPROVED = "approved"
OVERRIDDEN = "overridden"
ESCALATED = "escalated"
TIMED_OUT = "timed_out"
SHADOW = "shadow"
REJECTED = "rejected"

TimeoutAction = Literal["escalate", "deny", "queue"]
EnforcementMode = Literal["enforce", "shadow"]
HumanOutcome = Literal["approved", "overridden", "escalated"]
ExplicitOutcome = Literal["approved", "rejected"]
RecordedOutcome = Literal[
    "approved", "overridden", "escalated", "timed_out", "rejected"
]


def _assert_never(value: NoReturn) -> NoReturn:
    raise ReviewError(f"unhandled review variant: {value!r}")


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError(f"{field_name} is required")
    return value.strip()


def validate_timeout_action(on_timeout: str) -> TimeoutAction:
    if on_timeout == "escalate":
        return "escalate"
    if on_timeout == "deny":
        return "deny"
    if on_timeout == "queue":
        return "queue"
    raise ReviewError("on_timeout must be one of: escalate, deny, queue")


def validate_enforcement(enforcement: str) -> EnforcementMode:
    if enforcement == "enforce":
        return "enforce"
    if enforcement == "shadow":
        return "shadow"
    raise ReviewError("enforcement must be one of: enforce, shadow")


def validate_human_outcome(outcome: str) -> HumanOutcome:
    if outcome == APPROVED:
        return "approved"
    if outcome == OVERRIDDEN:
        return "overridden"
    if outcome == ESCALATED:
        return "escalated"
    raise ReviewError("outcome must be one of: approved, overridden, escalated")


def validate_explicit_outcome(outcome: str) -> ExplicitOutcome:
    if outcome == APPROVED:
        return "approved"
    if outcome == REJECTED:
        return "rejected"
    raise ReviewError("outcome must be one of: approved, rejected")


@dataclass(frozen=True)
class ReviewRequest:
    """Payload delivered to a reviewer or webhook."""

    review_id: str
    decision_id: str
    action: str
    summary: Dict[str, object]
    reviewer: str
    timeout_s: float
    on_timeout: str
    agent_id: str
    agent_name: str
    requested_at: str
    evidence_hash: str
    escalate_to: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewResolution:
    """A named human's decision on a pending review."""

    outcome: HumanOutcome
    reviewer: str
    reason: str = ""
    to_action: str = ""
    to_reviewer: str = ""

    def __post_init__(self) -> None:
        outcome = validate_human_outcome(self.outcome)
        reviewer = _require_text(self.reviewer, "reviewer")
        reason = self.reason.strip() if isinstance(self.reason, str) else ""
        if outcome in (OVERRIDDEN, ESCALATED) and not reason:
            raise ReviewError(f"{outcome} requires a non-empty reason")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "reviewer", reviewer)
        object.__setattr__(self, "reason", reason)


@dataclass
class ReviewDecision:
    """What ``require_review`` returns after the checkpoint finishes."""

    outcome: str
    reviewer: str
    reason: str
    decided_at: str
    evidence_hash: str
    prev_hash: str
    decision_id: str = ""
    review_id: str = ""
    timeout_action: str = ""
    to_action: str = ""
    to_reviewer: str = ""
    events: List[LayrEvent] = field(default_factory=list)


def decision_from_events(
    *,
    outcome: str,
    reviewer: str,
    reason: str,
    decision_id: str,
    review_id: str,
    events: Sequence[LayrEvent],
    timeout_action: str = "",
    to_action: str = "",
    to_reviewer: str = "",
) -> ReviewDecision:
    terminal = events[-1]
    metadata = terminal.metadata
    decided_at = metadata.get("decided_at", terminal.timestamp)
    evidence_hash = metadata.get("evidence_hash", "")
    prev_hash = metadata.get("prev_hash", "")
    return ReviewDecision(
        outcome=outcome,
        reviewer=reviewer,
        reason=reason,
        decided_at=decided_at if isinstance(decided_at, str) else terminal.timestamp,
        evidence_hash=evidence_hash if isinstance(evidence_hash, str) else "",
        prev_hash=prev_hash if isinstance(prev_hash, str) else "",
        decision_id=decision_id,
        review_id=review_id,
        timeout_action=timeout_action,
        to_action=to_action,
        to_reviewer=to_reviewer,
        events=list(events),
    )


ReviewHandler = Callable[
    [ReviewRequest],
    Union[ReviewResolution, None, Awaitable[Optional[ReviewResolution]]],
]
DeliverHook = Callable[[ReviewRequest], Union[None, Awaitable[None]]]


def sign_payload(secret: str, body: bytes) -> str:
    """Return the ``X-Layr-Signature`` value for a webhook body."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def request_body(request: ReviewRequest) -> bytes:
    return canonical_json(request.to_dict()).encode("utf-8")


class ReviewBackend(ABC):
    """Delivers a review request and waits for a named human."""

    @abstractmethod
    async def open_review(self, request: ReviewRequest) -> None:
        """Register the request and notify the reviewer."""

    @abstractmethod
    async def wait_for_resolution(
        self, review_id: str, timeout_s: float
    ) -> Optional[ReviewResolution]:
        """Return the human decision, or None when ``timeout_s`` elapses."""

    @abstractmethod
    def resolve(
        self, review_id: str, resolution: ReviewResolution, *, strict: bool = False
    ) -> None:
        """Record a decision. Safe to call from another thread.

        ``strict`` raises when the review is unknown or already decided.
        Webhook callbacks use strict mode. An in-memory ``resolve`` before
        ``open_review`` is remembered until the checkpoint starts.
        """

    def discard(self, review_id: str) -> None:
        """Drop a pending review after the checkpoint has returned."""

    async def aclose(self) -> None:
        """Release backend resources."""


class CallbackReviewBackend(ReviewBackend):
    """In-memory queue with an optional callback.

    If ``handler`` returns a ``ReviewResolution``, the checkpoint completes
    immediately. If it returns None, or there is no handler, call ``resolve``
    from another task or thread before ``timeout_s``.
    """

    def __init__(self, handler: Optional[ReviewHandler] = None) -> None:
        self._handler = handler
        self._lock = Lock()
        self._pending: Dict[str, asyncio.Future[ReviewResolution]] = {}
        self._requests: Dict[str, ReviewRequest] = {}
        self._early: Dict[str, ReviewResolution] = {}

    def pending_ids(self) -> List[str]:
        with self._lock:
            return [
                review_id for review_id, fut in self._pending.items() if not fut.done()
            ]

    async def open_review(self, request: ReviewRequest) -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ReviewResolution] = loop.create_future()
        with self._lock:
            early = self._early.pop(request.review_id, None)
            self._pending[request.review_id] = future
            self._requests[request.review_id] = request
        if early is not None:
            self.resolve(request.review_id, early)
        if self._handler is None:
            return
        result = self._handler(request)
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            if not isinstance(result, ReviewResolution):
                raise ReviewError("review handler must return ReviewResolution or None")
            self.resolve(request.review_id, result)

    async def wait_for_resolution(
        self, review_id: str, timeout_s: float
    ) -> Optional[ReviewResolution]:
        with self._lock:
            future = self._pending.get(review_id)
        if future is None:
            raise ReviewError(f"unknown review: {review_id}")
        if future.done():
            return future.result()
        if timeout_s <= 0:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout_s)
        except asyncio.TimeoutError:
            return None

    def resolve(
        self, review_id: str, resolution: ReviewResolution, *, strict: bool = False
    ) -> None:
        if not isinstance(resolution, ReviewResolution):
            raise ReviewError("resolution must be a ReviewResolution")
        with self._lock:
            future = self._pending.get(review_id)
            if future is None:
                if strict:
                    raise ReviewError(f"unknown review: {review_id}")
                self._early[review_id] = resolution
                return
        if future.done():
            if strict:
                raise ReviewError(f"review already decided: {review_id}")
            return
        loop = future.get_loop()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        def _set() -> None:
            if not future.done():
                future.set_result(resolution)

        if running is loop:
            _set()
            return
        loop.call_soon_threadsafe(_set)

    def discard(self, review_id: str) -> None:
        with self._lock:
            self._pending.pop(review_id, None)
            self._requests.pop(review_id, None)
            self._early.pop(review_id, None)


class WebhookReviewBackend(ReviewBackend):
    """Notify a webhook, then wait for ``handle_callback``.

    Pass ``deliver`` to send the request yourself (Slack, email, a test).
    Otherwise the backend POSTs canonical JSON to ``url`` and signs it with
    ``secret`` when one is set. The signature header is ``X-Layr-Signature:
    sha256=<hex>`` over the raw body.
    """

    def __init__(
        self,
        url: str = "",
        *,
        secret: str = "",
        deliver: Optional[DeliverHook] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        delivery_timeout_s: float = 10.0,
    ) -> None:
        if not url and deliver is None:
            raise ReviewError("webhook backend requires url or deliver hook")
        self._url = url
        self._secret = secret
        self._deliver = deliver
        self._client = http_client
        self._owns_client = False
        self._delivery_timeout_s = delivery_timeout_s
        self._inner = CallbackReviewBackend()

    def pending_ids(self) -> List[str]:
        return self._inner.pending_ids()

    async def open_review(self, request: ReviewRequest) -> None:
        await self._inner.open_review(request)
        try:
            await self._deliver_request(request)
        except Exception:
            self._inner.discard(request.review_id)
            raise

    async def wait_for_resolution(
        self, review_id: str, timeout_s: float
    ) -> Optional[ReviewResolution]:
        return await self._inner.wait_for_resolution(review_id, timeout_s)

    def resolve(
        self, review_id: str, resolution: ReviewResolution, *, strict: bool = False
    ) -> None:
        self._inner.resolve(review_id, resolution, strict=strict)

    def discard(self, review_id: str) -> None:
        self._inner.discard(review_id)

    def handle_callback(self, body: bytes, signature: str = "") -> ReviewResolution:
        """Apply a signed webhook callback and unblock ``require_review``."""
        if not isinstance(body, (bytes, bytearray)):
            raise ReviewError("webhook body must be bytes")
        raw = bytes(body)
        if self._secret:
            expected = sign_payload(self._secret, raw)
            if not signature or not hmac.compare_digest(expected, signature):
                raise ReviewError("invalid webhook signature")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewError("webhook body is not JSON") from exc
        if not isinstance(parsed, dict):
            raise ReviewError("webhook body must be a JSON object")
        review_id = parsed.get("review_id")
        if not isinstance(review_id, str) or not review_id:
            raise ReviewError("webhook body is missing review_id")
        resolution = _resolution_from_payload(parsed)
        self.resolve(review_id, resolution, strict=True)
        return resolution

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
            self._owns_client = False

    async def _deliver_request(self, request: ReviewRequest) -> None:
        try:
            if self._deliver is not None:
                result = self._deliver(request)
                if inspect.isawaitable(result):
                    await result
                return
            await self._post(request)
        except ReviewError:
            raise
        except Exception as exc:
            raise ReviewError(f"review webhook delivery failed: {exc}") from exc

    async def _post(self, request: ReviewRequest) -> None:
        if not self._url:
            raise ReviewError("webhook url is required")
        body = request_body(request)
        headers = {
            "Content-Type": "application/json",
            "X-Layr-Event": "review_requested",
        }
        if self._secret:
            headers["X-Layr-Signature"] = sign_payload(self._secret, body)
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self._delivery_timeout_s)
            self._client = client
            self._owns_client = True
        response = await client.post(self._url, content=body, headers=headers)
        response.raise_for_status()


def _resolution_from_payload(payload: Mapping[str, object]) -> ReviewResolution:
    outcome = payload.get("outcome", "")
    reviewer = payload.get("reviewer", "")
    reason = payload.get("reason", "")
    to_action = payload.get("to_action", "")
    to_reviewer = payload.get("to_reviewer", "")
    if not isinstance(outcome, str) or not isinstance(reviewer, str):
        raise ReviewError("webhook resolution requires string outcome and reviewer")
    return ReviewResolution(
        outcome=validate_human_outcome(outcome),
        reviewer=reviewer,
        reason=reason if isinstance(reason, str) else "",
        to_action=to_action if isinstance(to_action, str) else "",
        to_reviewer=to_reviewer if isinstance(to_reviewer, str) else "",
    )


def action_for_outcome(outcome: RecordedOutcome) -> str:
    """Map a terminal human outcome to the LayrEvent action_type."""
    if outcome == "approved":
        return "review_approved"
    if outcome == "overridden":
        return "review_overridden"
    if outcome == "escalated":
        return "review_escalated"
    if outcome == "timed_out":
        return "review_timed_out"
    if outcome == "rejected":
        return "review_rejected"
    _assert_never(outcome)


def evidence_kind_for_outcome(outcome: RecordedOutcome) -> str:
    if outcome == "approved" or outcome == "rejected" or outcome == "timed_out":
        return "human_review"
    if outcome == "overridden":
        return "override"
    if outcome == "escalated":
        return "escalation"
    _assert_never(outcome)
