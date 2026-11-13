import asyncio
import json
import threading
import time
from typing import Dict, List, Optional

import pytest

from layr import ReviewError
from layr.agent import Agent
from layr.evidence import verify_events
from layr.exporters.base import BaseExporter
from layr.exporters.otlp import set_layr_span_attributes
from layr.review import (
    CallbackReviewBackend,
    ReviewBackend,
    ReviewRequest,
    ReviewResolution,
    WebhookReviewBackend,
    sign_payload,
)
from layr.schema import LayrEvent


def _agent(**kwargs: object) -> Agent:
    config: Dict[str, object] = {
        "api_key": "k",
        "exporter": "local",
        "mode": "local",
        "fail_silent": False,
    }
    config.update(kwargs)
    return Agent(**config)  # type: ignore[arg-type]


def _approve(request: ReviewRequest) -> ReviewResolution:
    return ReviewResolution(
        outcome="approved",
        reviewer=request.reviewer,
        reason="Score within policy band",
    )


class CaptureExporter(BaseExporter):
    def __init__(self) -> None:
        self.events: List[LayrEvent] = []

    async def export(self, events):
        self.events.extend(events)


class FakeSpan:
    def __init__(self) -> None:
        self.attrs: Dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attrs[key] = value


class InstantBackend(ReviewBackend):
    def __init__(self) -> None:
        self.request: Optional[ReviewRequest] = None

    async def open_review(self, request: ReviewRequest) -> None:
        self.request = request

    async def wait_for_resolution(
        self, review_id: str, timeout_s: float
    ) -> ReviewResolution:
        return ReviewResolution(
            outcome="escalated",
            reviewer="cro@acme.com",
            reason="Edge case — policy silent",
            to_reviewer="cro@acme.com",
        )

    def resolve(
        self, review_id: str, resolution: ReviewResolution, *, strict: bool = False
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_require_review_approved_is_article_26_evidence() -> None:
    agent = _agent(review_backend=CallbackReviewBackend(_approve))
    decision = await agent.require_review(
        decision_id="loan_88421",
        action="approve_loan",
        summary={"amount": 12000, "score": 0.71},
        reviewer="ava.chen@acme.com",
        timeout_s=3600,
        on_timeout="escalate",
    )
    assert decision.outcome == "approved"
    assert decision.reviewer == "ava.chen@acme.com"
    assert decision.reason == "Score within policy band"
    assert decision.evidence_hash
    assert decision.prev_hash
    assert [event.action_type for event in decision.events] == [
        "review_requested",
        "review_approved",
    ]

    approved = decision.events[-1]
    assert approved.evidence_kind == "human_review"
    assert approved.decision_id == "loan_88421"
    assert approved.article_26 == "26(2)"
    assert approved.article_26_duty == "human_oversight"
    assert approved.metadata["counts_as_oversight"] is True
    assert approved.metadata["retention_days"] == 180
    mappings = approved.metadata["article_26_mappings"]
    assert [row["paragraph"] for row in mappings] == ["26(2)", "26(6)"]
    assert mappings[1]["duty"] == "log_retention"
    assert mappings[1]["label"] == "logs"
    assert decision.events[0].metadata["counts_as_oversight"] is False
    assert "evidence_hash" in approved.to_dict()

    check = verify_events(decision.events, expected_tip=agent.evidence_log.tip)
    assert check.ok
    assert agent.evidence_log.verify().ok

    raw = json.loads(json.dumps([event.to_dict() for event in decision.events]))
    assert verify_events(raw, expected_tip=agent.evidence_log.tip).ok
    raw[1]["metadata"]["outcome"] = "overridden"
    tampered = verify_events(raw)
    assert not tampered.ok

    span = FakeSpan()
    set_layr_span_attributes(span, approved)
    assert span.attrs["layr.action_type"] == "review_approved"
    assert span.attrs["layr.evidence_hash"] == approved.evidence_hash
    assert span.attrs["layr.article_26"] == "26(2)"
    assert span.attrs["layr.meta.outcome"] == "approved"
    assert span.attrs["layr.meta.retention_days"] == 180
    assert isinstance(span.attrs["layr.metadata"], str)
    assert "human_oversight" in str(span.attrs["layr.metadata_json"])
    await agent.aclose()


@pytest.mark.asyncio
async def test_override_callback_timeouts_and_shadow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def override(request: ReviewRequest) -> ReviewResolution:
        return ReviewResolution(
            outcome="overridden",
            reviewer=request.reviewer,
            reason="Adverse media hit — manual deny",
            to_action="deny_loan",
        )

    agent = _agent(review_backend=CallbackReviewBackend(override))
    decision = await agent.require_review(
        decision_id="loan_9",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
        on_timeout="deny",
    )
    assert decision.outcome == "overridden"
    assert decision.to_action == "deny_loan"
    assert decision.events[-1].evidence_kind == "override"
    output = capsys.readouterr().out
    assert "evidence=override" in output
    assert "article=26(2)" in output

    async def escalate(request: ReviewRequest) -> ReviewResolution:
        return ReviewResolution(
            outcome="escalated",
            reviewer="ava.chen@acme.com",
            reason="Edge case — policy silent",
            to_reviewer="cro@acme.com",
        )

    agent.review_backend = CallbackReviewBackend(escalate)
    escalated = await agent.require_review(
        decision_id="loan_10",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
    )
    assert escalated.outcome == "escalated"
    assert escalated.to_reviewer == "cro@acme.com"
    assert escalated.events[-1].action_type == "review_escalated"

    agent.review_backend = CallbackReviewBackend()
    denied = await agent.require_review(
        decision_id="loan_11",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
        timeout_s=0,
        on_timeout="deny",
    )
    assert denied.outcome == "timed_out"
    assert denied.timeout_action == "deny"
    assert denied.events[-1].action_type == "review_timed_out"
    assert denied.events[-1].action_success is False
    assert denied.events[-1].metadata["counts_as_oversight"] is False

    queued = await agent.require_review(
        decision_id="loan_12",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
        timeout_s=0.05,
        on_timeout="queue",
    )
    assert queued.outcome == "timed_out"
    assert queued.timeout_action == "queue"

    escalated_timeout = await agent.require_review(
        decision_id="loan_13",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
        timeout_s=0,
        on_timeout="escalate",
        escalate_to="cro@acme.com",
    )
    assert escalated_timeout.outcome == "escalated"
    assert escalated_timeout.timeout_action == "escalate"
    assert [event.action_type for event in escalated_timeout.events] == [
        "review_requested",
        "review_timed_out",
        "review_escalated",
    ]
    assert escalated_timeout.events[-1].metadata["automatic"] is True
    assert escalated_timeout.events[-1].metadata["counts_as_oversight"] is False

    calls = {"n": 0}

    def count(_request: ReviewRequest) -> None:
        calls["n"] += 1
        return None

    agent.review_backend = CallbackReviewBackend(count)
    shadow = await agent.require_review(
        decision_id="loan_14",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
        enforcement="shadow",
    )
    assert shadow.outcome == "shadow"
    assert calls["n"] == 0
    assert shadow.events[0].metadata["counts_as_oversight"] is False
    assert agent.evidence_log.verify().ok
    await agent.aclose()


@pytest.mark.asyncio
async def test_in_memory_resolve_and_explicit_events() -> None:
    backend = CallbackReviewBackend()
    agent = _agent(review_backend=backend)
    task = asyncio.create_task(
        agent.require_review(
            decision_id="loan_15",
            action="approve_loan",
            reviewer="ava.chen@acme.com",
            timeout_s=2,
            on_timeout="deny",
        )
    )
    review_id = ""
    for _ in range(50):
        pending = backend.pending_ids()
        if pending:
            review_id = pending[0]
            break
        await asyncio.sleep(0.01)
    assert review_id
    backend.resolve(
        review_id,
        ReviewResolution(
            outcome="approved", reviewer="ava.chen@acme.com", reason="checked"
        ),
    )
    decision = await task
    assert decision.outcome == "approved"
    assert decision.review_id == review_id

    approved = await agent.human_review(
        decision_id="loan_16",
        reviewer="ava.chen@acme.com",
        outcome="approved",
        reason="within band",
    )
    rejected = await agent.human_review(
        decision_id="loan_16",
        reviewer="ava.chen@acme.com",
        outcome="rejected",
        reason="outside policy",
    )
    overridden = await agent.override(
        decision_id="loan_16",
        reviewer="ava.chen@acme.com",
        from_action="approve_loan",
        to_action="deny_loan",
        reason="Adverse media hit — manual deny",
    )
    escalated = await agent.escalate(
        decision_id="loan_16",
        from_reviewer="ava.chen@acme.com",
        to_reviewer="cro@acme.com",
        reason="Edge case — policy silent",
    )
    retention = await agent.configure_retention(
        days=180, region="eu-west-1", immutable=True
    )
    short = await agent.configure_retention(days=30, region="eu-west-1")
    assert approved.action_type == "review_approved"
    assert rejected.action_type == "review_rejected"
    assert overridden.metadata["from_action"] == "approve_loan"
    assert escalated.metadata["to_reviewer"] == "cro@acme.com"
    assert retention.article_26 == "26(6)"
    assert retention.metadata["retention_meets_article_26_6"] is True
    assert retention.metadata["storage_region"] == "eu-west-1"
    assert short.metadata["retention_meets_article_26_6"] is False
    assert short.metadata["retention_days"] == 30

    tracked = await agent.track(action="approve_loan", target="loan_16", success=True)
    mixed = [
        decision.events[0],
        tracked,
        decision.events[1],
        approved,
        rejected,
        overridden,
        escalated,
        retention,
        short,
    ]
    check = verify_events(mixed, expected_tip=agent.evidence_log.tip)
    assert check.ok
    assert check.records == len(agent.evidence_log)
    await agent.aclose()


@pytest.mark.asyncio
async def test_validation_and_custom_backend() -> None:
    agent = _agent()
    with pytest.raises(ReviewError):
        await agent.require_review(
            decision_id=" ", action="approve_loan", reviewer="ava.chen@acme.com"
        )
    with pytest.raises(ReviewError):
        await agent.require_review(
            decision_id="loan",
            action="approve_loan",
            reviewer="ava.chen@acme.com",
            on_timeout="drop",
        )
    with pytest.raises(ReviewError):
        await agent.require_review(
            decision_id="loan",
            action="approve_loan",
            reviewer="ava.chen@acme.com",
            timeout_s=-1,
        )
    with pytest.raises(ReviewError):
        await agent.override(
            decision_id="loan",
            reviewer="ava.chen@acme.com",
            from_action="approve_loan",
            to_action="deny_loan",
            reason=" ",
        )
    with pytest.raises(ReviewError):
        await agent.human_review(
            decision_id="loan",
            reviewer="ava.chen@acme.com",
            outcome="rejected",
            reason="",
        )
    with pytest.raises(ReviewError):
        ReviewResolution(outcome="overridden", reviewer="ava.chen@acme.com", reason="")
    assert len(agent.evidence_log) == 0

    backend = InstantBackend()
    agent.review_backend = backend
    decision = await agent.require_review(
        decision_id="loan_20",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
    )
    assert decision.outcome == "escalated"
    assert backend.request is not None
    assert backend.request.reviewer == "ava.chen@acme.com"
    assert agent.evidence_log.verify().ok
    await agent.aclose()


def test_resolve_from_another_thread() -> None:
    backend = CallbackReviewBackend()
    agent = _agent(review_backend=backend)
    found: Dict[str, object] = {}

    def run() -> None:
        try:
            found["decision"] = agent.require_review_sync(
                decision_id="loan_thread",
                action="approve_loan",
                reviewer="ava.chen@acme.com",
                timeout_s=2,
                on_timeout="deny",
            )
        except Exception as exc:
            found["error"] = exc

    worker = threading.Thread(target=run)
    worker.start()
    review_id = ""
    for _ in range(100):
        pending = backend.pending_ids()
        if pending:
            review_id = pending[0]
            break
        time.sleep(0.01)
    assert review_id
    backend.resolve(
        review_id,
        ReviewResolution(
            outcome="approved",
            reviewer="ava.chen@acme.com",
            reason="checked from another thread",
        ),
    )
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert "error" not in found
    decision = found["decision"]
    assert getattr(decision, "outcome") == "approved"
    assert agent.evidence_log.verify().ok


def test_require_review_sync_and_running_loop() -> None:
    agent = _agent(review_backend=CallbackReviewBackend(_approve))
    first = agent.require_review_sync(
        decision_id="loan_sync",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
    )
    second = agent.require_review_sync(
        decision_id="loan_sync_2",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
    )
    assert first.outcome == "approved"
    assert second.outcome == "approved"
    assert agent.evidence_log.verify().ok


@pytest.mark.asyncio
async def test_sync_wrapper_refuses_a_running_loop() -> None:
    agent = _agent()
    with pytest.raises(ReviewError):
        agent.require_review_sync(
            decision_id="loan",
            action="approve_loan",
            reviewer="ava.chen@acme.com",
        )
    await agent.aclose()


@pytest.mark.asyncio
async def test_webhook_signature_and_delivery() -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, url, content, headers):
            self.calls.append((url, content, headers))
            return FakeResponse()

    client = FakeClient()
    posted = WebhookReviewBackend(
        url="https://reviews.example/hook", secret="whsec", http_client=client
    )
    request = ReviewRequest(
        review_id="rev_1",
        decision_id="loan_1",
        action="approve_loan",
        summary={"amount": 1},
        reviewer="ava.chen@acme.com",
        timeout_s=10,
        on_timeout="deny",
        agent_id="agent",
        agent_name="credit",
        requested_at="2026-11-06T00:00:00+00:00",
        evidence_hash="abc",
    )
    await posted.open_review(request)
    assert len(client.calls) == 1
    url, body, headers = client.calls[0]
    assert url == "https://reviews.example/hook"
    assert headers["X-Layr-Signature"] == sign_payload("whsec", body)
    assert json.loads(body)["decision_id"] == "loan_1"
    posted.discard("rev_1")

    seen: Dict[str, str] = {}

    def deliver(pending: ReviewRequest) -> None:
        seen["id"] = pending.review_id

    backend = WebhookReviewBackend(secret="whsec", deliver=deliver)
    agent = _agent(review_backend=backend)
    task = asyncio.create_task(
        agent.require_review(
            decision_id="loan_30",
            action="approve_loan",
            reviewer="ava.chen@acme.com",
            timeout_s=2,
            on_timeout="deny",
        )
    )
    review_id = ""
    for _ in range(50):
        pending_ids = backend.pending_ids()
        if pending_ids:
            review_id = pending_ids[0]
            break
        await asyncio.sleep(0.01)
    assert review_id == seen["id"]
    payload = {
        "review_id": review_id,
        "outcome": "overridden",
        "reviewer": "ava.chen@acme.com",
        "reason": "Adverse media hit — manual deny",
        "to_action": "deny_loan",
    }
    raw = json.dumps(payload).encode("utf-8")
    with pytest.raises(ReviewError):
        backend.handle_callback(raw, "sha256=deadbeef")
    with pytest.raises(ReviewError):
        backend.handle_callback(
            json.dumps({**payload, "reason": ""}).encode("utf-8"),
            sign_payload(
                "whsec", json.dumps({**payload, "reason": ""}).encode("utf-8")
            ),
        )
    backend.handle_callback(raw, sign_payload("whsec", raw))
    decision = await task
    assert decision.outcome == "overridden"
    assert decision.reason == "Adverse media hit — manual deny"

    class DownClient:
        async def post(self, url, content, headers):
            raise RuntimeError("webhook down")

    failing = WebhookReviewBackend(
        url="https://reviews.example/hook", http_client=DownClient()
    )
    agent.review_backend = failing
    with pytest.raises(ReviewError):
        await agent.require_review(
            decision_id="loan_31",
            action="approve_loan",
            reviewer="ava.chen@acme.com",
            timeout_s=1,
        )
    assert agent.evidence_log.verify().ok
    await agent.aclose()
    await posted.aclose()


@pytest.mark.asyncio
async def test_evidence_exports_through_the_existing_pipeline() -> None:
    capture = CaptureExporter()
    agent = _agent(exporter="local", mode="production", fail_silent=False)
    agent._exporter = capture
    decision = await agent.require_review(
        decision_id="loan_40",
        action="approve_loan",
        reviewer="ava.chen@acme.com",
        backend=CallbackReviewBackend(_approve),
    )
    await agent.flush()
    await agent.aclose()
    kinds = [event.action_type for event in capture.events]
    assert kinds == ["review_requested", "review_approved"]
    assert capture.events[-1].metadata["article_26"] == "26(2)"
    assert verify_events(capture.events, expected_tip=decision.evidence_hash).ok
