from __future__ import annotations

from typing import Optional, Sequence

import httpx

from layr.exporters.base import BaseExporter
from layr.schema import LayrEvent


class GrafanaExporter(BaseExporter):
    def __init__(self, endpoint: str, token: str) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._client: Optional[httpx.AsyncClient] = None

    async def export(self, events: Sequence[LayrEvent]) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        await self._client.post(
            f"{self._endpoint}/loki/api/v1/push",
            headers={"Authorization": f"Bearer {self._token}"},
            json={
                "streams": [
                    {
                        "stream": {"service": "layr-sdk"},
                        "values": [[str(i), str(event.to_dict())] for i, event in enumerate(events)],
                    }
                ]
            },
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
