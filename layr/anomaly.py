from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from layr.schema import LayrEvent


@dataclass
class Baselines:
    actions_per_hour: float = 50.0
    tokens_per_session: float = 10000.0
    cost_per_hour_usd: float = 5.0
    error_rate: float = 0.02


def compute_deviation(event: LayrEvent, baselines: Baselines) -> float:
    action_var = abs(event.actions_per_hour - baselines.actions_per_hour) / max(
        baselines.actions_per_hour, 1.0
    )
    cost_var = abs(event.cost_per_hour_usd - baselines.cost_per_hour_usd) / max(
        baselines.cost_per_hour_usd, 0.0001
    )
    error_var = abs(event.error_rate - baselines.error_rate) / max(
        baselines.error_rate, 0.0001
    )
    return round((action_var + cost_var + error_var) / 3.0, 6)


def is_anomalous(event: LayrEvent, baselines: Baselines, threshold: float = 1.0) -> bool:
    return compute_deviation(event, baselines) >= threshold
