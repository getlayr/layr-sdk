from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, List, Optional

from layr.anomaly import Baselines, compute_deviation, is_anomalous
from layr.event import build_event
from layr.exceptions import ConfigurationError, ExportError
from layr.exporters.base import BaseExporter
from layr.exporters.datadog import DatadogExporter
from layr.exporters.grafana import GrafanaExporter
from layr.exporters.layr_cloud import LayrCloudExporter
from layr.exporters.local import LocalExporter
from layr.exporters.otlp import OTLPExporter
from layr.schema import LayrEvent
from layr.session import SessionContext, SessionState, new_session
from layr.utils import env, generate_id


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
    ) -> None:
        resolved_api_key = api_key or env("LAYR_API_KEY")
        if not resolved_api_key:
            raise ConfigurationError("api_key is required")

        self.config = AgentConfig(
            api_key=resolved_api_key,
            agent_name=agent_name or env("LAYR_AGENT_NAME", "agent") or "agent",
            agent_version=agent_version or env("LAYR_AGENT_VERSION", "0.0.1") or "0.0.1",
            environment=environment or env("LAYR_ENVIRONMENT", "production") or "production",
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
            return OTLPExporter(otlp_endpoint or env("LAYR_OTLP_ENDPOINT", "http://localhost:4318/v1/traces") or "")
        if self.config.exporter == "datadog":
            return DatadogExporter(
                datadog_api_key or env("LAYR_DATADOG_API_KEY", "") or "",
                datadog_site or env("LAYR_DATADOG_SITE", "datadoghq.com") or "datadoghq.com",
            )
        if self.config.exporter == "grafana":
            return GrafanaExporter(
                grafana_endpoint or env("LAYR_GRAFANA_ENDPOINT", "http://localhost:3000") or "",
                grafana_token or env("LAYR_GRAFANA_TOKEN", "") or "",
            )
        return LayrCloudExporter(self.config.api_key)

    async def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    async def _run_worker(self) -> None:
        while not self._shutdown:
            batch: List[LayrEvent] = []
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
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
        metadata: Optional[Dict[str, object]] = None,
        prompt_version: str = "",
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
                error_rate=(self._session.error_count / max(self._session.action_count, 1)),
                deviation_from_baseline=0.0,
                metadata=metadata,
                model=model,
            )
            event.deviation_from_baseline = compute_deviation(event, self._baselines)
            self._session.total_cost += event.estimated_cost_usd

        await self._ensure_worker()
        if self._anomaly_detection and is_anomalous(event, self._baselines) and self._on_anomaly:
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
