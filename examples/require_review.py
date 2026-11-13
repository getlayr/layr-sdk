"""Pause a high-risk decision for a named reviewer and verify the evidence chain."""

import asyncio

from layr import Agent, CallbackReviewBackend, ReviewResolution, verify_events


def _approve(request):
    return ReviewResolution(
        outcome="approved",
        reviewer=request.reviewer,
        reason="Score within policy band",
    )


async def main() -> None:
    agent = Agent(
        api_key="your-key",
        agent_name="credit-decision-agent",
        exporter="local",
        mode="local",
        review_backend=CallbackReviewBackend(_approve),
    )
    decision = await agent.require_review(
        decision_id="loan_88421",
        action="approve_loan",
        summary={"amount": 12000, "score": 0.71, "model": "gpt-4o"},
        reviewer="ava.chen@acme.com",
        timeout_s=3600,
        on_timeout="escalate",
    )
    print(f"outcome={decision.outcome} hash={decision.evidence_hash}")
    check = verify_events(decision.events, expected_tip=agent.evidence_log.tip)
    print(f"chain_ok={check.ok} tip={check.tip}")
    await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
