from __future__ import annotations

from typing import Optional, Sequence

import httpx

from layr.exporters.base import BaseExporter
from layr.schema import LayrEvent


class DatadogExporter(BaseExporter):
    def __init__(self, api_key: str, site: str = "datadoghq.com") -> None:
        self._api_key = api_key
        self._endpoint = f"https://http-intake.logs.{site}/api/v2/logs"
        self._client: Optional[httpx.AsyncClient] = None

    async def export(self, events: Sequence[LayrEvent]) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        await self._client.post(
            self._endpoint,
            headers={"DD-API-KEY": self._api_key, "Content-Type": "application/json"},
            json=[e.to_dict() for e in events],
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
