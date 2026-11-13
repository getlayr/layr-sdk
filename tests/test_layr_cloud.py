from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from layr.agent import Agent, _LOGGED_EXPORTER_CHOICES
from layr.event import build_event
from layr.exceptions import ExportError
from layr.exporters.layr_cloud import LayrCloudExporter, resolve_layr_endpoint
from layr.exporters.local import LocalExporter
from layr.session import new_session


def _event():
    return build_event(
        agent_id="a",
        agent_name="n",
        agent_version="0.0.1",
        framework="custom",
        environment="dev",
        default_model="gpt-4o",
        session=new_session("u"),
        action_type="x",
        action_target="y",
        action_duration_ms=1,
        action_success=True,
        action_error="",
        intent="",
        reasoning="",
        confidence=1.0,
        tools_considered=[],
        tools_used=[],
        input_tokens=1,
        output_tokens=1,
        prompt_version="v",
        parent_agent_id="",
        handoff_count=0,
        delegation_depth=0,
        network_id="",
        actions_per_hour=1.0,
        cost_per_hour_usd=0.0,
        error_rate=0.0,
        deviation_from_baseline=0.0,
        metadata={},
        model=None,
    )


class _Response:
    def __init__(self, status_code: int, reason_phrase: str) -> None:
        self.status_code = status_code
        self.reason_phrase = reason_phrase


class _Client:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.urls: list[str] = []
        self.bodies: list[dict] = []

    async def post(self, url, json):
        self.urls.append(url)
        self.bodies.append(json)
        if self.error is not None:
            raise self.error
        return self.response

    async def aclose(self):
        return None


def test_endpoint_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAYR_ENDPOINT", "http://from-env")
    assert resolve_layr_endpoint("http://from-param") == "http://from-param/api/events"
    assert resolve_layr_endpoint(None) == "http://from-env/api/events"
    assert (
        LayrCloudExporter("k", endpoint="https://cloud.example/api/events/").endpoint
        == "https://cloud.example/api/events"
    )

    monkeypatch.delenv("LAYR_ENDPOINT")
    assert resolve_layr_endpoint(None) == "http://localhost:3000/api/events"
    assert resolve_layr_endpoint("") == "http://localhost:3000/api/events"

    agent = Agent(
        api_key="k",
        exporter="layr",
        endpoint="http://param-host",
    )
    assert isinstance(agent._exporter, LayrCloudExporter)
    assert agent._exporter.endpoint == "http://param-host/api/events"

    monkeypatch.setenv("LAYR_ENDPOINT", "http://from-env")
    from_env = Agent(api_key="k", exporter="layr")
    assert isinstance(from_env._exporter, LayrCloudExporter)
    assert from_env._exporter.endpoint == "http://from-env/api/events"


def test_api_key_selects_layr_and_local_mode_overrides(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _LOGGED_EXPORTER_CHOICES.clear()
    monkeypatch.setenv("LAYR_API_KEY", "env-key")
    monkeypatch.delenv("LAYR_EXPORTER", raising=False)
    monkeypatch.delenv("LAYR_ENDPOINT", raising=False)
    monkeypatch.delenv("LAYR_MODE", raising=False)
    with caplog.at_level(logging.INFO, logger="layr.agent"):
        implied = Agent()
        Agent()
        explicit = Agent(exporter="otlp")
        local = Agent(exporter="layr", mode="local")
    assert isinstance(implied._exporter, LayrCloudExporter)
    assert implied._exporter.endpoint == "http://localhost:3000/api/events"
    assert type(explicit._exporter).__name__ == "OTLPExporter"
    assert isinstance(local._exporter, LocalExporter)
    layr_logs = [
        record.message
        for record in caplog.records
        if "LAYR_API_KEY is set and no exporter was given" in record.message
    ]
    assert layr_logs == [
        "Layr exporter active: layr (LAYR_API_KEY is set and no exporter was given)."
    ]
    assert any(
        "LAYR_MODE=local overrides exporter='layr'" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_non_2xx_logs_and_strict_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = _event()
    client = _Client(_Response(401, "Unauthorized secret-key"))
    exporter = LayrCloudExporter("secret-key", endpoint="http://localhost:3000")
    exporter._client = client
    with caplog.at_level(logging.WARNING, logger="layr.exporters.layr_cloud"):
        await exporter.export([event])
    assert client.urls == ["http://localhost:3000/api/events"]
    assert client.bodies[0]["api_key"] == "secret-key"
    warning = caplog.records[-1]
    assert warning.levelno == logging.WARNING
    assert "status=401" in warning.message
    assert "reason=Unauthorized [redacted]" in warning.message
    assert "endpoint=http://localhost:3000/api/events" in warning.message
    assert "secret-key" not in warning.message

    strict = LayrCloudExporter(
        "secret-key",
        endpoint="http://localhost:3000/api/events",
        raise_on_export_error=True,
    )
    strict._client = _Client(_Response(503, "unavailable"))
    with pytest.raises(ExportError) as raised:
        await strict.export([event])
    assert "status=503" in str(raised.value)
    assert "secret-key" not in str(raised.value)


@pytest.mark.asyncio
async def test_connection_error_logs_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exporter = LayrCloudExporter(
        "secret-key", endpoint="https://api.example/api/events"
    )
    exporter._client = _Client(
        error=httpx.ConnectError("connection refused secret-key")
    )
    with caplog.at_level(logging.WARNING, logger="layr.exporters.layr_cloud"):
        await exporter.export([_event()])
    message = caplog.records[-1].message
    assert "status=connection_error" in message
    assert "ConnectError" in message
    assert "endpoint=https://api.example/api/events" in message
    assert "secret-key" not in message


@pytest.mark.asyncio
async def test_agent_strict_flag_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    agent = Agent(
        api_key="secret-key",
        exporter="layr",
        endpoint="http://example.test",
        raise_on_export_error=True,
        fail_silent=True,
    )
    assert isinstance(agent._exporter, LayrCloudExporter)
    agent._exporter._client = _Client(_Response(500, "Server Error secret-key"))
    with pytest.raises(ExportError) as raised:
        await agent._send_with_retry([_event()])
    assert "status=500" in str(raised.value)
    assert "secret-key" not in str(raised.value)

    quiet = Agent(
        api_key="secret-key",
        exporter="layr",
        endpoint="http://example.test",
        raise_on_export_error=False,
    )
    assert isinstance(quiet._exporter, LayrCloudExporter)
    quiet._exporter._client = _Client(_Response(500, "Server Error secret-key"))
    await quiet._send_with_retry([_event()])
