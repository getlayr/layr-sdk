# Changelog

## 0.2.0

- Added `Agent.require_review` / `require_review_sync` to pause a high-risk decision until a named human approves, overrides, or escalates.
- Added in-memory callback and webhook review backends, plus explicit `human_review`, `override`, and `escalate` events.
- Added a hash-chained evidence log (`verify_events`, `EvidenceLog.verify`) and Article 26 duty metadata on those events.
- Evidence events export through the existing OTLP, Datadog, Grafana, Layr Cloud, and local exporters. OTLP spans keep `layr.<field>` attributes and add `layr.meta.*` plus `layr.metadata_json`.
- Layr Cloud endpoint is configurable with `endpoint=` or `LAYR_ENDPOINT` (default `http://localhost:3000/api/events`). A base URL gets `/api/events` appended.
- Layr Cloud HTTP and connection failures log a warning and do not raise unless `raise_on_export_error=True`.
- When `LAYR_API_KEY` is set and no exporter is given, the SDK selects the Layr Cloud exporter and logs the active exporter once. `LAYR_MODE=local` still forces the local exporter.

## 0.1.0

- Rebuilt SDK as `layr-sdk` with typed observability schema.
- Added async batching, retry, local mode, anomaly detection.
- Added exporters (OTLP, Datadog, Grafana, Layr Cloud, Local).
- Added framework integrations (LangChain, CrewAI, AutoGen wrappers).
