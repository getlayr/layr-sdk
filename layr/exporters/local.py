from __future__ import annotations

from typing import Sequence

from layr.exporters.base import BaseExporter
from layr.schema import LayrEvent


class LocalExporter(BaseExporter):
    async def export(self, events: Sequence[LayrEvent]) -> None:
        for event in events:
            lines = (
                "[LAYR] agent={agent}\n"
                "       action={action}\n"
                "       target={target}\n"
                "       tokens={tokens} cost=${cost}\n"
                "       latency={latency}ms\n"
                "       success={success}"
            ).format(
                agent=event.agent_name,
                action=event.action_type,
                target=event.action_target,
                tokens=event.total_tokens,
                cost=event.estimated_cost_usd,
                latency=event.model_latency_ms,
                success=event.action_success,
            )
            if event.evidence_hash:
                lines += (
                    "\n       evidence={kind} article={article} hash={digest}".format(
                        kind=event.evidence_kind,
                        article=event.article_26,
                        digest=event.evidence_hash[:12],
                    )
                )
            print(lines)
