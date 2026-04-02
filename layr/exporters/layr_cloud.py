from __future__ import annotations

from typing import Optional, Sequence

import httpx

from layr.exporters.base import BaseExporter
from layr.schema import LayrEvent


class LayrCloudExporter(BaseExporter):
    def __init__(self, api_key: str, endpoint: str = "http://localhost:3000") -> None:
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def export(self, events: Sequence[LayrEvent]) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        for event in events:
            await self._client.post(
                f"{self._endpoint}/api/events",
                json={
                    "api_key": self._api_key,
                    "event": event.to_dict(),
                },
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
