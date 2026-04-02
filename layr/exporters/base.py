from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from layr.schema import LayrEvent


class BaseExporter(ABC):
    @abstractmethod
    async def export(self, events: Sequence[LayrEvent]) -> None:
        """Export a batch of events."""

    async def close(self) -> None:
        """Close any exporter resources."""
