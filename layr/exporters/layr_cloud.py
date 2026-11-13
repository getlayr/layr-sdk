from __future__ import annotations

import logging
from typing import Optional, Sequence

import httpx

from layr.exceptions import ExportError
from layr.exporters.base import BaseExporter
from layr.schema import LayrEvent
from layr.utils import env

logger = logging.getLogger("layr.exporters.layr_cloud")

DEFAULT_LAYR_ENDPOINT = "http://localhost:3000/api/events"
_EVENTS_SUFFIX = "/api/events"
_REASON_LIMIT = 200


def resolve_layr_endpoint(endpoint: Optional[str] = None) -> str:
    """Pick the events URL. An explicit value beats LAYR_ENDPOINT, then the default.

    A host or base URL such as ``http://localhost:3000`` has ``/api/events``
    appended. A value that already ends with that path is kept as-is.
    """
    chosen = endpoint if endpoint else env("LAYR_ENDPOINT")
    if not chosen:
        chosen = DEFAULT_LAYR_ENDPOINT
    return _ensure_events_path(chosen)


def _ensure_events_path(url: str) -> str:
    trimmed = url.strip().rstrip("/")
    if trimmed.lower().endswith(_EVENTS_SUFFIX):
        return trimmed
    return f"{trimmed}{_EVENTS_SUFFIX}"


def _short_reason(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _REASON_LIMIT:
        return collapsed
    return collapsed[: _REASON_LIMIT - 1] + "…"


class LayrCloudExporter(BaseExporter):
    def __init__(
        self,
        api_key: str,
        endpoint: Optional[str] = None,
        *,
        raise_on_export_error: bool = False,
    ) -> None:
        self._api_key = api_key
        self._endpoint = resolve_layr_endpoint(endpoint)
        self._raise_on_export_error = raise_on_export_error
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def export(self, events: Sequence[LayrEvent]) -> None:
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            self._client = client
        for event in events:
            await self._post_event(client, event)

    async def _post_event(self, client: httpx.AsyncClient, event: LayrEvent) -> None:
        payload = {"api_key": self._api_key, "event": event.to_dict()}
        try:
            response = await client.post(self._endpoint, json=payload)
        except httpx.HTTPError as exc:
            self._fail("connection_error", f"{type(exc).__name__}: {exc}")
            return
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            return
        if 200 <= int(status_code) < 300:
            return
        reason = getattr(response, "reason_phrase", "") or ""
        self._fail(str(int(status_code)), str(reason))

    def _fail(self, status: str, reason: str) -> None:
        safe_reason = self._redact(_short_reason(reason) or "request failed")
        safe_endpoint = self._redact(self._endpoint)
        logger.warning(
            "Layr export failed status=%s reason=%s endpoint=%s",
            status,
            safe_reason,
            safe_endpoint,
        )
        if self._raise_on_export_error:
            raise ExportError(
                "Layr export failed "
                f"status={status} reason={safe_reason} endpoint={safe_endpoint}"
            )

    def _redact(self, text: str) -> str:
        if not self._api_key:
            return text
        return text.replace(self._api_key, "[redacted]")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
