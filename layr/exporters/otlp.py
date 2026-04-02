from __future__ import annotations

from typing import Sequence

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from layr.exporters.base import BaseExporter
from layr.schema import LayrEvent


class OTLPExporter(BaseExporter):
    def __init__(self, endpoint: str) -> None:
        provider = TracerProvider(resource=Resource.create({"service.name": "layr-sdk"}))
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("layr")

    async def export(self, events: Sequence[LayrEvent]) -> None:
        for event in events:
            with self._tracer.start_as_current_span(event.action_type) as span:
                for key, value in event.to_dict().items():
                    span.set_attribute(f"layr.{key}", str(value))
