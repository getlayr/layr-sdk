"""Tamper-evident hash chain for oversight evidence.

Each record is::

    evidence_hash = sha256(prev_hash + canonical_json(payload) + timestamp)

``canonical_json`` is UTF-8 JSON with sorted keys and no insignificant
whitespace (``separators=(",", ":")``, ``ensure_ascii=False``). The first
record links to ``GENESIS_HASH`` (64 zero hex characters). ``timestamp`` is
the event's UTC ISO-8601 timestamp and is also stored inside the payload.

The chain detects edits and deletions. It is not a signature: a party who
rewrites every later hash produces a different tip. Hold on to the tip
(``EvidenceLog.tip`` or ``expected_tip=``) to detect a full rewrite.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from layr.exceptions import EvidenceError
from layr.schema import LayrEvent

GENESIS_HASH = "0" * 64

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List["JsonValue"], Dict[str, "JsonValue"]]


def _canonicalize(value: object) -> JsonValue:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError("evidence payload contains a non-finite number")
        return value
    if isinstance(value, dict):
        normalized: Dict[str, JsonValue] = {}
        for key in sorted(value.keys(), key=str):
            if not isinstance(key, str):
                raise EvidenceError("evidence payload keys must be strings")
            normalized[key] = _canonicalize(value[key])
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise EvidenceError(f"unsupported evidence value type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Deterministic JSON used as the hash preimage."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_evidence_hash(
    prev_hash: str, payload: Mapping[str, object], timestamp: str
) -> str:
    """sha256(prev_hash || canonical_json(payload) || timestamp)."""
    if not isinstance(prev_hash, str) or not prev_hash:
        raise EvidenceError("prev_hash must be a non-empty string")
    if not isinstance(timestamp, str) or not timestamp:
        raise EvidenceError("timestamp must be a non-empty string")
    material = prev_hash + canonical_json(payload) + timestamp
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceRecord:
    index: int
    timestamp: str
    prev_hash: str
    evidence_hash: str
    payload: Dict[str, object]


@dataclass(frozen=True)
class EvidenceVerification:
    """Result of checking a chain.

    ``ok`` is true when every link recomputes and, when ``expected_tip`` was
    provided, the final hash matches it. ``broken_index`` is the first record
    that failed.
    """

    ok: bool
    records: int
    tip: str
    broken_index: Optional[int] = None
    reason: str = ""


def _fail(index: int, seen: int, reason: str) -> EvidenceVerification:
    return EvidenceVerification(
        ok=False,
        records=seen,
        tip="",
        broken_index=index,
        reason=reason,
    )


def verify_evidence_chain(
    records: Sequence[EvidenceRecord],
    *,
    genesis: str = GENESIS_HASH,
    expected_tip: Optional[str] = None,
) -> EvidenceVerification:
    """Recompute a hash chain.

    A record whose ``index`` is 0 must link to ``genesis``. A later slice
    (first index > 0) is checked internally against its own ``prev_hash``;
    pass ``expected_tip`` from a tip you stored earlier to detect rewrites.
    """
    if not records:
        tip = genesis
        if expected_tip is not None and expected_tip != tip:
            return _fail(0, 0, "chain tip mismatch")
        return EvidenceVerification(ok=True, records=0, tip=tip)

    prev = genesis if records[0].index == 0 else records[0].prev_hash
    for offset, record in enumerate(records):
        if offset == 0 and record.index == 0 and record.prev_hash != genesis:
            return _fail(record.index, len(records), "prev_hash mismatch")
        if offset > 0:
            previous = records[offset - 1]
            if record.index != previous.index + 1:
                return _fail(record.index, len(records), "evidence index gap")
            if record.prev_hash != prev:
                return _fail(record.index, len(records), "prev_hash mismatch")
        elif record.prev_hash != prev:
            return _fail(record.index, len(records), "prev_hash mismatch")
        try:
            expected = compute_evidence_hash(
                record.prev_hash, record.payload, record.timestamp
            )
        except EvidenceError as exc:
            return _fail(record.index, len(records), str(exc))
        if expected != record.evidence_hash:
            return _fail(record.index, len(records), "evidence_hash mismatch")
        stamped = record.payload.get("timestamp")
        if stamped is not None and stamped != record.timestamp:
            return _fail(record.index, len(records), "timestamp mismatch")
        indexed = record.payload.get("evidence_index")
        if indexed is not None and indexed != record.index:
            return _fail(record.index, len(records), "evidence_index mismatch")
        prev = record.evidence_hash

    if expected_tip is not None and prev != expected_tip:
        return _fail(records[-1].index, len(records), "chain tip mismatch")
    return EvidenceVerification(ok=True, records=len(records), tip=prev)


class EvidenceLog:
    """Append-only hash chain kept on the agent that emitted the events."""

    def __init__(self, genesis: str = GENESIS_HASH) -> None:
        self._genesis = genesis
        self._tip = genesis
        self._records: List[EvidenceRecord] = []
        self._lock = Lock()

    @property
    def tip(self) -> str:
        with self._lock:
            return self._tip

    @property
    def genesis(self) -> str:
        return self._genesis

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def records(self) -> Tuple[EvidenceRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def append(self, payload: Mapping[str, object], timestamp: str) -> EvidenceRecord:
        """Hash ``payload`` onto the chain.

        ``evidence_index`` is written into the hashed payload so a deleted
        row cannot be renumbered without changing the tip.
        """
        with self._lock:
            index = len(self._records)
            body: Dict[str, object] = dict(payload)
            body["evidence_index"] = index
            if "timestamp" in body and body["timestamp"] != timestamp:
                raise EvidenceError(
                    "payload timestamp does not match evidence timestamp"
                )
            body["timestamp"] = timestamp
            canonical = _canonicalize(body)
            if not isinstance(canonical, dict):
                raise EvidenceError("evidence payload must be an object")
            prev = self._tip if self._records else self._genesis
            digest = compute_evidence_hash(prev, canonical, timestamp)
            record = EvidenceRecord(
                index=index,
                timestamp=timestamp,
                prev_hash=prev,
                evidence_hash=digest,
                payload=dict(canonical),
            )
            self._records.append(record)
            self._tip = digest
            return record

    def verify(self, *, expected_tip: Optional[str] = None) -> EvidenceVerification:
        snapshot = self.records()
        tip = expected_tip if expected_tip is not None else self.tip
        return verify_evidence_chain(snapshot, genesis=self._genesis, expected_tip=tip)


def _as_mapping(value: object) -> Optional[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return value
    return None


def _event_dict(event: object) -> Optional[Dict[str, object]]:
    if isinstance(event, LayrEvent):
        return event.to_dict()
    mapping = _as_mapping(event)
    if mapping is None:
        return None
    return dict(mapping)


def records_from_events(
    events: Sequence[object],
) -> Tuple[List[EvidenceRecord], Optional[EvidenceVerification]]:
    """Pull evidence records out of Layr events or exported dicts.

    Events without ``evidence_hash`` are skipped so a normal ``track`` call
    can sit in the same batch. Returns ``(records, error)``.
    """
    records: List[EvidenceRecord] = []
    for event in events:
        data = _event_dict(event)
        if data is None:
            return [], _fail(0, 0, "event is not a LayrEvent or object")
        metadata = _as_mapping(data.get("metadata"))
        if metadata is None:
            continue
        digest = metadata.get("evidence_hash", data.get("evidence_hash", ""))
        if digest in ("", None):
            continue
        payload = metadata.get("evidence_payload")
        payload_map = _as_mapping(payload)
        if payload_map is None or not isinstance(digest, str):
            return [], _fail(
                len(records),
                len(records) + 1,
                "evidence event is missing payload or hashes",
            )
        prev = metadata.get("prev_hash", data.get("prev_hash", ""))
        if not isinstance(prev, str) or not prev:
            return [], _fail(
                len(records), len(records) + 1, "evidence event is missing prev_hash"
            )
        timestamp = data.get("timestamp", "")
        if not isinstance(timestamp, str) or not timestamp:
            return [], _fail(
                len(records), len(records) + 1, "evidence event is missing timestamp"
            )
        index_value = payload_map.get("evidence_index", metadata.get("evidence_index"))
        if not isinstance(index_value, int) or isinstance(index_value, bool):
            return [], _fail(
                len(records),
                len(records) + 1,
                "evidence event is missing evidence_index",
            )
        if payload_map.get("action_type") != data.get("action_type"):
            return [], _fail(
                index_value,
                len(records) + 1,
                "action_type does not match evidence payload",
            )
        if metadata.get("outcome") != payload_map.get("outcome"):
            return [], _fail(
                index_value, len(records) + 1, "outcome does not match evidence payload"
            )
        if metadata.get("evidence_kind") != payload_map.get("evidence_kind"):
            return [], _fail(
                index_value,
                len(records) + 1,
                "evidence_kind does not match evidence payload",
            )
        top_hash = data.get("evidence_hash")
        if isinstance(top_hash, str) and top_hash and top_hash != digest:
            return [], _fail(
                index_value,
                len(records) + 1,
                "evidence_hash does not match event field",
            )
        decided_at = metadata.get("decided_at")
        if isinstance(decided_at, str) and decided_at and decided_at != timestamp:
            return [], _fail(
                index_value, len(records) + 1, "decided_at does not match timestamp"
            )
        records.append(
            EvidenceRecord(
                index=index_value,
                timestamp=timestamp,
                prev_hash=prev,
                evidence_hash=digest,
                payload=dict(payload_map),
            )
        )
    return records, None


def verify_events(
    events: Sequence[object],
    *,
    expected_tip: Optional[str] = None,
    genesis: str = GENESIS_HASH,
) -> EvidenceVerification:
    """Verify evidence events exported through any Layr exporter.

    Accepts ``LayrEvent`` instances or ``to_dict`` / JSON objects. Non-evidence
    events in the batch are ignored.
    """
    records, error = records_from_events(events)
    if error is not None:
        return error
    return verify_evidence_chain(records, genesis=genesis, expected_tip=expected_tip)
