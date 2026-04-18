"""
Output Validator — post-LLM checks: hallucination grounding, toxicity, schema compliance.
"""

import re
from typing import Any

from src.models.schemas import GuardrailDecision, GuardrailResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Toxicity Patterns ──────────────────────────────────────────────────────────
TOXIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(idiot|moron|stupid\s+user|shut\s+up|you\s+suck)\b", re.IGNORECASE),
    re.compile(r"(?i)(kill\s+yourself|kys|go\s+die)"),
]

# ── Hallucination Indicators ───────────────────────────────────────────────────
HALLUCINATION_INDICATORS = [
    re.compile(r"(?i)as\s+of\s+my\s+knowledge\s+cutoff"),
    re.compile(r"(?i)I\s+believe\s+but\s+(am\s+not\s+sure|cannot\s+verify)"),
    re.compile(r"(?i)(made-up|fictional|hypothetical)\s+example"),
]

# ── Overconfidence Patterns ────────────────────────────────────────────────────
OVERCONFIDENCE = [
    re.compile(r"(?i)this\s+is\s+(100%|absolutely|definitely)\s+(true|correct|accurate)"),
    re.compile(r"(?i)I\s+(guarantee|promise)\s+this\s+is"),
]


def check_toxicity(text: str) -> list[str]:
    """Detect toxic language patterns."""
    found = []
    for p in TOXIC_PATTERNS:
        match = p.search(text)
        if match:
            found.append(match.group(0))
    return found


def check_hallucination_risk(text: str, context: str = "") -> dict[str, Any]:
    """
    Estimate hallucination risk:
    - Detects explicit uncertainty markers
    - If context provided, checks that key claims appear in context
    """
    indicators = [p.pattern for p in HALLUCINATION_INDICATORS if p.search(text)]
    overconfident = [p.pattern for p in OVERCONFIDENCE if p.search(text)]

    risk_score = 0.0
    if indicators:
        risk_score += 0.3
    if overconfident:
        risk_score += 0.4

    # Simple grounding check: count content words in text that also appear in context
    if context:
        text_words = set(re.findall(r"\b\w{5,}\b", text.lower()))
        ctx_words = set(re.findall(r"\b\w{5,}\b", context.lower()))
        if text_words:
            overlap = len(text_words & ctx_words) / len(text_words)
            if overlap < 0.2:
                risk_score = min(risk_score + 0.4, 1.0)

    return {
        "risk_score": round(risk_score, 2),
        "uncertainty_markers": indicators,
        "overconfident_claims": overconfident,
        "is_high_risk": risk_score >= 0.6,
    }


def validate_output(
    text: str,
    context: str = "",
    block_toxic: bool = True,
    block_high_hallucination: bool = False,
) -> GuardrailResult:
    """
    Full output validation pipeline.

    Args:
        text:                   Generated LLM response text.
        context:                Retrieved context (for grounding check).
        block_toxic:            If True, block toxic responses.
        block_high_hallucination: If True, block responses with high hallucination risk.

    Returns:
        GuardrailResult.
    """
    issues = []

    # Toxicity check
    toxic = check_toxicity(text)
    if toxic and block_toxic:
        logger.warning(f"OutputValidator: toxic content detected: {toxic}")
        return GuardrailResult(
            decision=GuardrailDecision.BLOCK,
            reason="Toxic content detected in response",
            detected_issues=toxic,
        )
    if toxic:
        issues.extend(toxic)

    # Hallucination risk check
    hallucination = check_hallucination_risk(text, context)
    if hallucination["is_high_risk"] and block_high_hallucination:
        logger.warning(f"OutputValidator: high hallucination risk: {hallucination['risk_score']}")
        return GuardrailResult(
            decision=GuardrailDecision.BLOCK,
            reason=f"High hallucination risk (score={hallucination['risk_score']})",
            detected_issues=["hallucination_risk"],
        )

    if issues:
        return GuardrailResult(
            decision=GuardrailDecision.SANITIZE,
            reason="Minor issues detected",
            detected_issues=issues,
        )

    return GuardrailResult(decision=GuardrailDecision.ALLOW)
