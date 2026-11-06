from __future__ import annotations

import json
from typing import Protocol, Sequence, Union

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from layr.exporters.base import BaseExporter
from layr.schema import LayrEvent

AttributeValue = Union[str, bool, int, float]


class SpanAttributes(Protocol):
    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Record one span attribute."""


def set_layr_span_attributes(span: SpanAttributes, event: LayrEvent) -> None:
    """Copy a Layr event onto a span without changing the existing attribute names.

    Top-level fields stay ``layr.<field>`` string attributes. Scalar metadata
    keys are also set as ``layr.meta.<key>`` so evidence fields (outcome,
    reviewer, retention, article duty) can be filtered. ``layr.metadata_json``
    is the metadata object as JSON.
    """
    data = event.to_dict()
    metadata = data.get("metadata")
    for key, value in data.items():
        span.set_attribute(f"layr.{key}", _span_string(value))
    if isinstance(metadata, dict):
        span.set_attribute(
            "layr.metadata_json", json.dumps(metadata, sort_keys=True, default=str)
        )
        for key, value in metadata.items():
            if isinstance(value, bool) or isinstance(value, (int, float, str)):
                span.set_attribute(f"layr.meta.{key}", value)


def _span_string(value: object) -> str:
    return str(value)


class OTLPExporter(BaseExporter):
    def __init__(self, endpoint: str) -> None:
        provider = TracerProvider(
            resource=Resource.create({"service.name": "layr-sdk"})
        )
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer("layr")

    async def export(self, events: Sequence[LayrEvent]) -> None:
        for event in events:
            with self._tracer.start_as_current_span(event.action_type) as span:
                set_layr_span_attributes(span, event)
