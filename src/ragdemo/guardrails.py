"""Small, explicit guardrails. These are baseline checks, not a complete security solution."""
from __future__ import annotations

import re

from .retrieval import Hit

# Below this best dense cosine similarity, the docs probably don't cover the question. Calibrated with eval/run_eval.py on 30 in-domain + 8 off-topic questions (in-domain min 0.694, off-topic max 0.646); a small set, so re-check it if you change the corpus or model.
MIN_RELEVANCE = 0.67

_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)"
    r"|(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)"
    r"|you\s+are\s+now\s+(?!able)",
    re.IGNORECASE,
)

REFUSAL = "I couldn't find this in the FastAPI tutorial docs I was given, so I won't guess. Try rephrasing, or ask about a topic the tutorial covers."
INJECTION_REFUSAL = "That looks like an attempt to change my instructions. I only answer questions about the FastAPI tutorial docs."


def looks_like_injection(question: str) -> bool:
    return bool(_INJECTION.search(question))


def is_relevant(hits: list[Hit], threshold: float = MIN_RELEVANCE) -> bool:
    return bool(hits) and max(h.dense for h in hits) >= threshold
