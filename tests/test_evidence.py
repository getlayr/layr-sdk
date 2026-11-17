import hashlib
import json

import pytest

from layr.evidence import (
    GENESIS_HASH,
    EvidenceLog,
    EvidenceRecord,
    canonical_json,
    compute_evidence_hash,
    verify_evidence_chain,
    verify_events,
)
from layr.exceptions import EvidenceError
from layr.event import build_event
from layr.session import new_session


def test_canonical_json_and_hash_preimage() -> None:
    left = {"b": 1, "a": {"d": 1, "c": [2, 1]}}
    right = {"a": {"c": [2, 1], "d": 1}, "b": 1}
    assert canonical_json(left) == canonical_json(right)
    timestamp = "2026-11-06T00:00:00+00:00"
    digest = compute_evidence_hash(GENESIS_HASH, left, timestamp)
    preimage = GENESIS_HASH + canonical_json(left) + timestamp
    assert digest == hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    assert len(GENESIS_HASH) == 64


def test_chain_detects_edits_gaps_and_rewrites() -> None:
    log = EvidenceLog()
    first_at = "2026-11-06T00:00:00+00:00"
    second_at = "2026-11-06T00:00:01+00:00"
    third_at = "2026-11-06T00:00:02+00:00"
    log.append({"action_type": "review_requested", "outcome": "requested"}, first_at)
    log.append({"action_type": "review_approved", "outcome": "approved"}, second_at)
    log.append({"action_type": "review_escalated", "outcome": "escalated"}, third_at)
    original_tip = log.tip
    assert log.verify().ok
    assert log.verify().tip == original_tip

    records = list(log.records())
    tampered = dict(records[1].payload)
    tampered["outcome"] = "overridden"
    edited = EvidenceRecord(
        index=records[1].index,
        timestamp=records[1].timestamp,
        prev_hash=records[1].prev_hash,
        evidence_hash=records[1].evidence_hash,
        payload=tampered,
    )
    broken = verify_evidence_chain((records[0], edited, records[2]))
    assert not broken.ok
    assert broken.broken_index == 1
    assert broken.reason == "evidence_hash mismatch"

    skipped = verify_evidence_chain((records[0], records[2]))
    assert not skipped.ok
    assert skipped.reason == "evidence index gap"

    first_payload = dict(records[0].payload)
    first_payload["reason"] = "rewritten"
    forged_first = compute_evidence_hash(
        GENESIS_HASH, first_payload, records[0].timestamp
    )
    second_payload = dict(records[1].payload)
    forged_second = compute_evidence_hash(
        forged_first, second_payload, records[1].timestamp
    )
    third_payload = dict(records[2].payload)
    forged_third = compute_evidence_hash(
        forged_second, third_payload, records[2].timestamp
    )
    rewritten = (
        EvidenceRecord(
            0, records[0].timestamp, GENESIS_HASH, forged_first, first_payload
        ),
        EvidenceRecord(
            1, records[1].timestamp, forged_first, forged_second, second_payload
        ),
        EvidenceRecord(
            2, records[2].timestamp, forged_second, forged_third, third_payload
        ),
    )
    assert verify_evidence_chain(rewritten).ok
    caught = verify_evidence_chain(rewritten, expected_tip=original_tip)
    assert not caught.ok
    assert caught.reason == "chain tip mismatch"


def test_evidence_payload_must_be_json_data() -> None:
    log = EvidenceLog()
    with pytest.raises(EvidenceError):
        log.append({"when": object()}, "2026-11-06T00:00:00+00:00")
    with pytest.raises(EvidenceError):
        log.append({"score": float("nan")}, "2026-11-06T00:00:00+00:00")


def test_normal_event_dict_keeps_previous_shape() -> None:
    event = build_event(
        agent_id="agent",
        agent_name="agent",
        agent_version="0.0.1",
        framework="custom",
        environment="dev",
        default_model="gpt-4o",
        session=new_session("user"),
        action_type="send_email",
        action_target="target",
        action_duration_ms=100,
        action_success=True,
        action_error="",
        intent="intent",
        reasoning="reasoning",
        confidence=0.5,
        tools_considered=["a"],
        tools_used=["a"],
        input_tokens=100,
        output_tokens=50,
        prompt_version="v1",
        parent_agent_id="",
        handoff_count=0,
        delegation_depth=0,
        network_id="",
        actions_per_hour=10.0,
        cost_per_hour_usd=0.0,
        error_rate=0.0,
        deviation_from_baseline=0.0,
        metadata={"k": "v"},
        model="gpt-4o",
    )
    assert "evidence_hash" not in event.to_dict()
    assert event.evidence_hash == ""
    exported = json.loads(json.dumps(event.to_dict()))
    assert verify_events([exported]).ok
    assert verify_events([exported]).records == 0
