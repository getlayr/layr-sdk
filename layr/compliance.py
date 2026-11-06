"""Article 26 duty labels carried on evidence events.

Paragraph numbers follow the duty map in the live-evidence spec, which matches
Regulation (EU) 2024/1689 Article 26:

- 26(2) human oversight
- 26(5) monitoring and incidents
- 26(6) logs kept under deployer control (at least six months)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Duty = Tuple[str, str]

# Primary paragraph for each evidence_kind the SDK can emit.
EVIDENCE_KIND_DUTIES: Dict[str, Duty] = {
    "human_review": ("26(2)", "human_oversight"),
    "override": ("26(2)", "human_oversight"),
    "escalation": ("26(2)", "human_oversight"),
    "incident": ("26(5)", "monitor_incidents"),
    "input_check": ("26(4)", "input_data"),
    "disclosure": ("26(11)", "inform_natural_persons"),
    "retention": ("26(6)", "log_retention"),
}

# Every hash-chained evidence record is itself a retained log.
LOG_DUTY: Duty = ("26(6)", "log_retention")

DUTY_LABELS: Dict[str, str] = {
    "human_oversight": "human oversight",
    "monitor_incidents": "monitor and incidents",
    "input_data": "input data",
    "inform_natural_persons": "inform natural persons",
    "log_retention": "logs",
}

ArticleMapping = Dict[str, str]


def article_mappings(evidence_kind: str) -> List[ArticleMapping]:
    """Return the Article 26 rows an evidence event attests.

    The first row is the primary duty. Review events also carry 26(6) because
    the hash-chained record is the retained log of that oversight decision.
    """
    try:
        paragraph, name = EVIDENCE_KIND_DUTIES[evidence_kind]
    except KeyError as exc:
        raise ValueError(f"unknown evidence kind: {evidence_kind}") from exc
    primary: ArticleMapping = {
        "paragraph": paragraph,
        "duty": name,
        "label": DUTY_LABELS.get(name, name),
        "role": "primary",
    }
    if (paragraph, name) == LOG_DUTY:
        return [primary]
    log_paragraph, log_name = LOG_DUTY
    return [
        primary,
        {
            "paragraph": log_paragraph,
            "duty": log_name,
            "label": DUTY_LABELS[log_name],
            "role": "log",
        },
    ]
