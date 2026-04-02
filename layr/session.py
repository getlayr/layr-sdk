from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from layr.utils import generate_id, utc_now_iso

if TYPE_CHECKING:
    from layr.agent import Agent


@dataclass
class SessionState:
    session_id: str
    session_start: str
    triggered_by: str
    action_count: int = 0
    error_count: int = 0
    total_cost: float = 0.0


def new_session(triggered_by: str) -> SessionState:
    return SessionState(
        session_id=generate_id(),
        session_start=utc_now_iso(),
        triggered_by=triggered_by,
    )


class SessionContext:
    def __init__(self, agent: "Agent", triggered_by: str) -> None:
        self._agent = agent
        self._triggered_by = triggered_by

    def __enter__(self) -> "Agent":
        self._agent.session_start(triggered_by=self._triggered_by)
        return self._agent

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._agent.session_end()
