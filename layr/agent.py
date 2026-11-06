from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, List, Mapping, Optional, Union, cast

from layr.anomaly import Baselines, compute_deviation, is_anomalous
from layr.compliance import article_mappings
from layr.event import build_event
from layr.evidence import EvidenceLog
from layr.exceptions import ConfigurationError, EvidenceError, ExportError, ReviewError
from layr.exporters.base import BaseExporter
from layr.exporters.datadog import DatadogExporter
from layr.exporters.grafana import GrafanaExporter
from layr.exporters.layr_cloud import LayrCloudExporter
from layr.exporters.local import LocalExporter
from layr.exporters.otlp import OTLPExporter
from layr.review import (
    ESCALATED,
    OVERRIDDEN,
    REJECTED,
    SHADOW,
    TIMED_OUT,
    CallbackReviewBackend,
    EnforcementMode,
    RecordedOutcome,
    ReviewBackend,
    ReviewDecision,
    ReviewRequest,
    ReviewResolution,
    TimeoutAction,
    _assert_never,
    action_for_outcome,
    decision_from_events,
    evidence_kind_for_outcome,
    validate_enforcement,
    validate_explicit_outcome,
    validate_timeout_action,
    _require_text,
)
from layr.schema import LayrEvent
from layr.session import SessionContext, SessionState, new_session
from layr.utils import env, generate_id, utc_now_iso


@dataclass
class AgentConfig:
    api_key: str
    agent_name: str
    agent_version: str
    environment: str
    framework: str
    model: str
    exporter: str
    mode: str


class Agent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        agent_name: Optional[str] = None,
        environment: Optional[str] = None,
        framework: Optional[str] = None,
        model: Optional[str] = None,
        exporter: Optional[str] = None,
        mode: Optional[str] = None,
        agent_version: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        datadog_api_key: Optional[str] = None,
        datadog_site: Optional[str] = None,
        grafana_endpoint: Optional[str] = None,
        grafana_token: Optional[str] = None,
        anomaly_detection: bool = False,
        baselines: Optional[Dict[str, float]] = None,
        on_anomaly: Optional[Callable[[LayrEvent], None]] = None,
        fail_silent: bool = True,
        review_backend: Optional[ReviewBackend] = None,
        retention_days: int = 180,
        storage_region: str = "",
        retention_immutable: bool = True,
    ) -> None:
        resolved_api_key = api_key or env("LAYR_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError("api_key is required")

        self.config = AgentConfig(
            api_key=resolved_api_key,
            agent_name=agent_name or env("LAYR_AGENT_NAME", "agent") or "agent",
            agent_version=agent_version
            or env("LAYR_AGENT_VERSION", "0.0.1")
            or "0.0.1",
            environment=environment
            or env("LAYR_ENVIRONMENT", "production")
            or "production",
            framework=framework or env("LAYR_FRAMEWORK", "custom") or "custom",
            model=model or env("LAYR_MODEL", "gpt-4o") or "gpt-4o",
            exporter=exporter or env("LAYR_EXPORTER", "otlp") or "otlp",
            mode=mode or env("LAYR_MODE", "production") or "production",
        )
        self.agent_id = generate_id()
        self._session: SessionState = new_session("sdk_init")
        self._lock = Lock()
        self._queue: "asyncio.Queue[LayrEvent]" = asyncio.Queue(maxsize=5000)
        self._batch_size = 100
        self._flush_interval = 5.0
        self._worker: Optional[asyncio.Task[None]] = None
        self._shutdown = False
        self._fail_silent = fail_silent
        self._anomaly_detection = anomaly_detection
        self._on_anomaly = on_anomaly
        self._baselines = Baselines(
            actions_per_hour=(baselines or {}).get("actions_per_hour", 50.0),
            tokens_per_session=(baselines or {}).get("tokens_per_session", 10000.0),
            cost_per_hour_usd=(baselines or {}).get("cost_per_hour_usd", 5.0),
            error_rate=(baselines or {}).get("error_rate", 0.02),
        )
        self._network_id = ""
        self._parent_agent_id = ""
        self._handoff_count = 0
        self._delegation_depth = 0
        self._worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self._evidence_log = EvidenceLog()
        self.review_backend: ReviewBackend = review_backend or CallbackReviewBackend()
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, int)
            or retention_days < 0
        ):
            raise ConfigurationError("retention_days must be a non-negative integer")
        if not isinstance(storage_region, str):
            raise ConfigurationError("storage_region must be a string")
        self._retention_days = retention_days
        self._storage_region = storage_region
        self._retention_immutable = bool(retention_immutable)
        self._exporter = self._make_exporter(
            otlp_endpoint=otlp_endpoint,
            datadog_api_key=datadog_api_key,
            datadog_site=datadog_site,
            grafana_endpoint=grafana_endpoint,
            grafana_token=grafana_token,
        )

    def _make_exporter(
        self,
        *,
        otlp_endpoint: Optional[str],
        datadog_api_key: Optional[str],
        datadog_site: Optional[str],
        grafana_endpoint: Optional[str],
        grafana_token: Optional[str],
    ) -> BaseExporter:
        if self.config.mode == "local" or self.config.exporter == "local":
            return LocalExporter()
        if self.config.exporter == "otlp":
            return OTLPExporter(
                otlp_endpoint
                or env("LAYR_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
                or ""
            )
        if self.config.exporter == "datadog":
            return DatadogExporter(
                datadog_api_key or env("LAYR_DATADOG_API_KEY", "") or "",
                datadog_site
                or env("LAYR_DATADOG_SITE", "datadoghq.com")
                or "datadoghq.com",
            )
        if self.config.exporter == "grafana":
            return GrafanaExporter(
                grafana_endpoint
                or env("LAYR_GRAFANA_ENDPOINT", "http://localhost:3000")
                or "",
                grafana_token or env("LAYR_GRAFANA_TOKEN", "") or "",
            )
        return LayrCloudExporter(self.config.api_key)

    async def _ensure_worker(self) -> None:
        loop = asyncio.get_running_loop()
        if self._worker_loop is not None and self._worker_loop is not loop:
            if self._worker is not None and not self._worker.done():
                raise ExportError("agent is already running on another event loop")
            self._queue = asyncio.Queue(maxsize=5000)
            self._worker = None
        self._worker_loop = loop
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    async def _run_worker(self) -> None:
        while not self._shutdown:
            batch: List[LayrEvent] = []
            try:
                first = await asyncio.wait_for(
                    self._queue.get(), timeout=self._flush_interval
                )
                batch.append(first)
            except asyncio.TimeoutError:
                continue
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            await self._send_with_retry(batch)

    async def _send_with_retry(self, events: List[LayrEvent]) -> None:
        delay = 0.1
        for attempt in range(5):
            try:
                await self._exporter.export(events)
                return
            except Exception as exc:
                if attempt == 4:
                    if self._fail_silent:
                        return
                    raise ExportError(str(exc))
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)

    def session_start(self, triggered_by: str) -> None:
        with self._lock:
            self._session = new_session(triggered_by)

    def session_end(self) -> None:
        with self._lock:
            self._session = new_session("session_end")

    def session(self, triggered_by: str) -> SessionContext:
        return SessionContext(self, triggered_by)

    async def track(
        self,
        action: str,
        target: str = "",
        intent: str = "",
        reasoning: str = "",
        confidence: float = 1.0,
        tools_considered: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int = 0,
        model: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Mapping[str, object]] = None,
        prompt_version: str = "",
        timestamp: Optional[str] = None,
    ) -> LayrEvent:
        with self._lock:
            self._session.action_count += 1
            if not success:
                self._session.error_count += 1

            actions_per_hour = float(self._session.action_count)
            event = build_event(
                agent_id=self.agent_id,
                agent_name=self.config.agent_name,
                agent_version=self.config.agent_version,
                framework=self.config.framework,
                environment=self.config.environment,
                default_model=self.config.model,
                session=self._session,
                action_type=action,
                action_target=target,
                action_duration_ms=latency_ms,
                action_success=success,
                action_error="" if success else "action_failed",
                intent=intent,
                reasoning=reasoning,
                confidence=confidence,
                tools_considered=tools_considered,
                tools_used=tools_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_version=prompt_version,
                parent_agent_id=self._parent_agent_id,
                handoff_count=self._handoff_count,
                delegation_depth=self._delegation_depth,
                network_id=self._network_id,
                actions_per_hour=actions_per_hour,
                cost_per_hour_usd=self._session.total_cost,
                error_rate=(
                    self._session.error_count / max(self._session.action_count, 1)
                ),
                deviation_from_baseline=0.0,
                metadata=metadata,
                model=model,
                timestamp=timestamp,
            )
            event.deviation_from_baseline = compute_deviation(event, self._baselines)
            self._session.total_cost += event.estimated_cost_usd

        await self._ensure_worker()
        if (
            self._anomaly_detection
            and is_anomalous(event, self._baselines)
            and self._on_anomaly
        ):
            self._on_anomaly(event)

        if self.config.mode == "local":
            await self._exporter.export([event])
            return event

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            if not self._fail_silent:
                raise ExportError("event queue is full")
        return event

    @property
    def evidence_log(self) -> EvidenceLog:
        """Hash-chained oversight log for this agent."""
        return self._evidence_log

    async def require_review(
        self,
        decision_id: str,
        action: str,
        summary: Optional[Dict[str, object]] = None,
        reviewer: str = "",
        timeout_s: float = 3600,
        on_timeout: str = "escalate",
        *,
        escalate_to: str = "",
        enforcement: str = "enforce",
        backend: Optional[ReviewBackend] = None,
        review_id: Optional[str] = None,
    ) -> ReviewDecision:
        """Pause a high-risk decision until a named human resolves it.

        Emits ``review_requested`` immediately, then one of ``review_approved``,
        ``review_overridden``, ``review_escalated``, or ``review_timed_out``.
        A timeout with ``on_timeout="escalate"`` emits both the timeout and the
        escalation. ``enforcement="shadow"`` records the request and returns
        without waiting, so the call is not oversight evidence.

        Use ``require_review_sync`` from code that is not already async.
        """
        return await self._require_review(
            decision_id=decision_id,
            action=action,
            summary=summary,
            reviewer=reviewer,
            timeout_s=timeout_s,
            on_timeout=on_timeout,
            escalate_to=escalate_to,
            enforcement=enforcement,
            backend=backend,
            review_id=review_id,
        )

    def require_review_sync(
        self,
        decision_id: str,
        action: str,
        summary: Optional[Dict[str, object]] = None,
        reviewer: str = "",
        timeout_s: float = 3600,
        on_timeout: str = "escalate",
        *,
        escalate_to: str = "",
        enforcement: str = "enforce",
        backend: Optional[ReviewBackend] = None,
        review_id: Optional[str] = None,
    ) -> ReviewDecision:
        """Blocking wrapper around ``require_review`` for sync callers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._require_review(
                    decision_id=decision_id,
                    action=action,
                    summary=summary,
                    reviewer=reviewer,
                    timeout_s=timeout_s,
                    on_timeout=on_timeout,
                    escalate_to=escalate_to,
                    enforcement=enforcement,
                    backend=backend,
                    review_id=review_id,
                )
            )
        raise ReviewError(
            "require_review_sync() was called from a running event loop; await agent.require_review()"
        )

    async def human_review(
        self,
        decision_id: str,
        reviewer: str,
        outcome: str,
        reason: str = "",
        summary: Optional[Dict[str, object]] = None,
    ) -> LayrEvent:
        """Record an explicit approve or reject without pausing."""
        explicit = validate_explicit_outcome(outcome)
        if explicit == REJECTED and not reason.strip():
            raise ReviewError("rejected requires a non-empty reason")
        recorded = cast(RecordedOutcome, explicit)
        return await self._emit_evidence(
            action_type=action_for_outcome(recorded),
            evidence_kind=evidence_kind_for_outcome(recorded),
            decision_id=_require_text(decision_id, "decision_id"),
            outcome=explicit,
            reviewer=_require_text(reviewer, "reviewer"),
            reason=reason.strip(),
            summary=summary,
            intent="human_review",
            counts_as_oversight=True,
        )

    async def override(
        self,
        decision_id: str,
        reviewer: str,
        from_action: str,
        to_action: str,
        reason: str,
        summary: Optional[Dict[str, object]] = None,
    ) -> LayrEvent:
        """Record a named human replacing one action with another."""
        if not reason.strip():
            raise ReviewError("override requires a non-empty reason")
        return await self._emit_evidence(
            action_type="review_overridden",
            evidence_kind="override",
            decision_id=_require_text(decision_id, "decision_id"),
            outcome=OVERRIDDEN,
            reviewer=_require_text(reviewer, "reviewer"),
            reason=reason.strip(),
            summary=summary,
            intent=_require_text(to_action, "to_action"),
            counts_as_oversight=True,
            extra={
                "from_action": _require_text(from_action, "from_action"),
                "to_action": _require_text(to_action, "to_action"),
            },
        )

    async def escalate(
        self,
        decision_id: str,
        from_reviewer: str,
        to_reviewer: str,
        reason: str,
        summary: Optional[Dict[str, object]] = None,
    ) -> LayrEvent:
        """Record a handoff from one named reviewer to another."""
        if not reason.strip():
            raise ReviewError("escalation requires a non-empty reason")
        source = _require_text(from_reviewer, "from_reviewer")
        target = _require_text(to_reviewer, "to_reviewer")
        return await self._emit_evidence(
            action_type="review_escalated",
            evidence_kind="escalation",
            decision_id=_require_text(decision_id, "decision_id"),
            outcome=ESCALATED,
            reviewer=source,
            reason=reason.strip(),
            summary=summary,
            intent="escalate",
            counts_as_oversight=True,
            extra={"from_reviewer": source, "to_reviewer": target},
        )

    async def configure_retention(
        self,
        days: int = 180,
        region: str = "",
        immutable: bool = True,
    ) -> LayrEvent:
        """Set Article 26(6) retention metadata and emit a retention event.

        ``days`` below 180 is stored and flagged ``retention_meets_article_26_6``
        false. The regulation's floor for deployer-controlled logs is six months.
        """
        if isinstance(days, bool) or not isinstance(days, int) or days < 0:
            raise ReviewError("retention days must be a non-negative integer")
        if not isinstance(region, str):
            raise ReviewError("storage region must be a string")
        self._retention_days = days
        self._storage_region = region
        self._retention_immutable = bool(immutable)
        return await self._emit_evidence(
            action_type="retention_configured",
            evidence_kind="retention",
            decision_id="retention",
            outcome="configured",
            reviewer="",
            reason="",
            intent="configure_retention",
            counts_as_oversight=False,
            extra={"retention_meets_article_26_6": days >= 180},
        )

    async def _require_review(
        self,
        *,
        decision_id: str,
        action: str,
        summary: Optional[Dict[str, object]],
        reviewer: str,
        timeout_s: float,
        on_timeout: str,
        escalate_to: str,
        enforcement: str,
        backend: Optional[ReviewBackend],
        review_id: Optional[str],
    ) -> ReviewDecision:
        decision_key = _require_text(decision_id, "decision_id")
        action_name = _require_text(action, "action")
        reviewer_name = _require_text(reviewer, "reviewer")
        timeout_action = validate_timeout_action(on_timeout)
        mode = validate_enforcement(enforcement)
        timeout_value = _timeout_seconds(timeout_s)
        if summary is not None and not isinstance(summary, dict):
            raise ReviewError("summary must be an object")
        review_key = (
            review_id.strip()
            if isinstance(review_id, str) and review_id.strip()
            else generate_id()
        )
        started = time.monotonic()
        common = {
            "action": action_name,
            "review_id": review_key,
            "timeout_s": timeout_value,
            "on_timeout": timeout_action,
            "enforcement": mode,
            "escalate_to": escalate_to,
        }
        requested = await self._emit_evidence(
            action_type="review_requested",
            evidence_kind="human_review",
            decision_id=decision_key,
            outcome="requested",
            reviewer=reviewer_name,
            reason="",
            summary=summary,
            intent=action_name,
            counts_as_oversight=False,
            extra={**common, "review_latency_ms": 0},
        )
        events = [requested]
        if mode == "shadow":
            return decision_from_events(
                outcome=SHADOW,
                reviewer=reviewer_name,
                reason="",
                decision_id=decision_key,
                review_id=review_key,
                events=events,
            )

        active = backend or self.review_backend
        request = ReviewRequest(
            review_id=review_key,
            decision_id=decision_key,
            action=action_name,
            summary=dict(summary or {}),
            reviewer=reviewer_name,
            timeout_s=float(timeout_value),
            on_timeout=timeout_action,
            agent_id=self.agent_id,
            agent_name=self.config.agent_name,
            requested_at=requested.timestamp,
            evidence_hash=requested.evidence_hash,
            escalate_to=escalate_to,
        )
        try:
            await active.open_review(request)
            resolution = await active.wait_for_resolution(
                review_key, float(timeout_value)
            )
        except Exception:
            active.discard(review_key)
            raise
        active.discard(review_key)

        latency_ms = max(0, int((time.monotonic() - started) * 1000))
        if resolution is None:
            return await self._timeout_decision(
                decision_id=decision_key,
                action=action_name,
                summary=summary,
                reviewer=reviewer_name,
                timeout_action=timeout_action,
                timeout_s=timeout_value,
                escalate_to=escalate_to,
                mode=mode,
                review_id=review_key,
                requested_at=requested.timestamp,
                latency_ms=latency_ms,
                events=events,
            )

        outcome = resolution.outcome
        extra: Dict[str, object] = {
            **common,
            "requested_at": requested.timestamp,
            "review_latency_ms": latency_ms,
            "from_action": action_name,
        }
        to_action = ""
        to_reviewer = ""
        if outcome == "overridden":
            to_action = resolution.to_action
            extra["to_action"] = to_action
        elif outcome == "escalated":
            to_reviewer = resolution.to_reviewer or escalate_to
            extra["from_reviewer"] = reviewer_name
            extra["to_reviewer"] = to_reviewer
        elif outcome == "approved":
            pass
        else:
            _assert_never(outcome)
        event = await self._emit_evidence(
            action_type=action_for_outcome(outcome),
            evidence_kind=evidence_kind_for_outcome(outcome),
            decision_id=decision_key,
            outcome=outcome,
            reviewer=resolution.reviewer,
            reason=resolution.reason,
            summary=summary,
            intent=action_name,
            counts_as_oversight=True,
            extra=extra,
        )
        events.append(event)
        return decision_from_events(
            outcome=outcome,
            reviewer=resolution.reviewer,
            reason=resolution.reason,
            decision_id=decision_key,
            review_id=review_key,
            events=events,
            to_action=to_action,
            to_reviewer=to_reviewer,
        )

    async def _timeout_decision(
        self,
        *,
        decision_id: str,
        action: str,
        summary: Optional[Dict[str, object]],
        reviewer: str,
        timeout_action: TimeoutAction,
        timeout_s: Union[int, float],
        escalate_to: str,
        mode: EnforcementMode,
        review_id: str,
        requested_at: str,
        latency_ms: int,
        events: List[LayrEvent],
    ) -> ReviewDecision:
        timeout_reason = f"review timed out after {timeout_s}s ({timeout_action})"
        base_extra: Dict[str, object] = {
            "action": action,
            "review_id": review_id,
            "timeout_s": timeout_s,
            "on_timeout": timeout_action,
            "enforcement": mode,
            "escalate_to": escalate_to,
            "requested_at": requested_at,
            "review_latency_ms": latency_ms,
        }
        timed_out = await self._emit_evidence(
            action_type="review_timed_out",
            evidence_kind="human_review",
            decision_id=decision_id,
            outcome=TIMED_OUT,
            reviewer=reviewer,
            reason=timeout_reason,
            summary=summary,
            success=False,
            intent=action,
            counts_as_oversight=False,
            automatic=True,
            extra=base_extra,
        )
        events.append(timed_out)
        if timeout_action == "escalate":
            escalate_reason = f"review timed out after {timeout_s}s and was escalated"
            escalated = await self._emit_evidence(
                action_type="review_escalated",
                evidence_kind="escalation",
                decision_id=decision_id,
                outcome=ESCALATED,
                reviewer=reviewer,
                reason=escalate_reason,
                summary=summary,
                intent=action,
                counts_as_oversight=False,
                automatic=True,
                extra={
                    **base_extra,
                    "from_reviewer": reviewer,
                    "to_reviewer": escalate_to,
                },
            )
            events.append(escalated)
            return decision_from_events(
                outcome=ESCALATED,
                reviewer=reviewer,
                reason=escalate_reason,
                decision_id=decision_id,
                review_id=review_id,
                events=events,
                timeout_action="escalate",
                to_reviewer=escalate_to,
            )
        if timeout_action == "deny" or timeout_action == "queue":
            return decision_from_events(
                outcome=TIMED_OUT,
                reviewer=reviewer,
                reason=timeout_reason,
                decision_id=decision_id,
                review_id=review_id,
                events=events,
                timeout_action=timeout_action,
            )
        _assert_never(timeout_action)

    async def _emit_evidence(
        self,
        *,
        action_type: str,
        evidence_kind: str,
        decision_id: str,
        outcome: str,
        reviewer: str,
        reason: str,
        summary: Optional[Dict[str, object]] = None,
        success: bool = True,
        intent: str = "",
        counts_as_oversight: bool = False,
        automatic: bool = False,
        extra: Optional[Dict[str, object]] = None,
    ) -> LayrEvent:
        if summary is None:
            summary_value: Dict[str, object] = {}
        elif isinstance(summary, dict):
            summary_value = dict(summary)
        else:
            raise ReviewError("summary must be an object")
        timestamp = utc_now_iso()
        try:
            mappings = article_mappings(evidence_kind)
        except ValueError as exc:
            raise ReviewError(str(exc)) from exc
        primary = mappings[0]
        with self._lock:
            session_id = self._session.session_id
        payload: Dict[str, object] = {}
        if extra:
            payload.update(extra)
        payload.update(
            {
                "action_type": action_type,
                "evidence_kind": evidence_kind,
                "decision_id": decision_id,
                "outcome": outcome,
                "reviewer": reviewer,
                "reason": reason,
                "agent_id": self.agent_id,
                "agent_name": self.config.agent_name,
                "session_id": session_id,
                "summary": summary_value,
                "timestamp": timestamp,
                "article_26": primary["paragraph"],
                "article_26_duty": primary["duty"],
                "article_26_mappings": mappings,
                "retention_days": self._retention_days,
                "storage_region": self._storage_region,
                "retention_immutable": self._retention_immutable,
                "counts_as_oversight": counts_as_oversight,
                "automatic": automatic,
            }
        )
        try:
            record = self._evidence_log.append(payload, timestamp)
        except EvidenceError as exc:
            raise ReviewError(str(exc)) from exc
        metadata: Dict[str, object] = dict(record.payload)
        metadata["prev_hash"] = record.prev_hash
        metadata["evidence_hash"] = record.evidence_hash
        metadata["decided_at"] = timestamp
        metadata["evidence_payload"] = record.payload
        return await self.track(
            action=action_type,
            target=decision_id,
            intent=intent or action_type,
            reasoning=reason,
            success=success,
            metadata=metadata,
            timestamp=timestamp,
        )

    async def flush(self) -> None:
        pending: List[LayrEvent] = []
        while True:
            try:
                pending.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if pending:
            await self._send_with_retry(pending)

    async def aclose(self) -> None:
        self._shutdown = True
        await self.flush()
        await self._exporter.close()
        await self.review_backend.aclose()


def _timeout_seconds(value: float) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewError("timeout_s must be a non-negative number")
    if isinstance(value, int):
        if value < 0:
            raise ReviewError("timeout_s must be a non-negative number")
        return value
    if value < 0 or value != value or value == float("inf") or value == float("-inf"):
        raise ReviewError("timeout_s must be a non-negative number")
    return float(value)


class AgentNetwork:
    def __init__(self, api_key: str) -> None:
        self.network_id = generate_id()
        self.api_key = api_key

    def agent(self, name: str) -> Agent:
        a = Agent(api_key=self.api_key, agent_name=name)
        a._network_id = self.network_id
        return a

    async def track_handoff(
        self,
        from_agent: Agent,
        to_agent: Agent,
        task: str,
        context: Optional[Dict[str, object]] = None,
    ) -> None:
        to_agent._parent_agent_id = from_agent.agent_id
        to_agent._handoff_count += 1
        to_agent._delegation_depth += 1
        await from_agent.track(
            action="handoff",
            target=to_agent.config.agent_name,
            intent=task,
            reasoning="delegated_task",
            metadata=context or {},
        )
