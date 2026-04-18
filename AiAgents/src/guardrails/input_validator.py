"""
Input Validator — pre-LLM safety checks: PII detection, prompt injection, content safety.
"""

import re
from typing import Any

from src.models.schemas import GuardrailDecision, GuardrailResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── PII Patterns ───────────────────────────────────────────────────────────────
PII_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b"),
    "password": re.compile(r"(?i)password\s*[:=]\s*\S+"),
    "api_key": re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{20,}"),
}

# ── Injection Patterns ─────────────────────────────────────────────────────────
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions?"),
    re.compile(r"(?i)disregard\s+(your\s+)?(system\s+)?prompt"),
    re.compile(r"(?i)you\s+are\s+now\s+(a|an)\s+"),
    re.compile(r"(?i)act\s+as\s+(if\s+)?you\s+(have\s+no\s+)?restrictions"),
    re.compile(r"(?i)forget\s+(everything|all)\s+you\s+(know|were\s+told)"),
    re.compile(r"(?i)jailbreak|DAN\s+mode|developer\s+mode"),
    re.compile(r"(?i)pretend\s+(to\s+be|you\s+are)\s+(an?\s+)?(?:evil|unrestricted)"),
]

# ── Harmful Content Keywords ───────────────────────────────────────────────────
HARMFUL_KEYWORDS = {
    "bomb", "explosive", "hack", "malware", "ransomware",
    "suicide\s+method", "self\s+harm", "buy\s+drugs",
}


def detect_pii(text: str) -> list[str]:
    """Return list of PII types found in the text."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
    return found


def detect_injection(text: str) -> bool:
    """Return True if prompt injection patterns are detected."""
    return any(p.search(text) for p in INJECTION_PATTERNS)


def detect_harmful_content(text: str) -> list[str]:
    """Return list of harmful keywords found."""
    found = []
    for keyword in HARMFUL_KEYWORDS:
        if re.search(keyword, text, re.IGNORECASE):
            found.append(keyword.replace("\\s+", " "))
    return found


def redact_pii(text: str) -> str:
    """Replace detected PII with [REDACTED TYPE] placeholders."""
    redacted = text
    replacements = {
        "ssn": "[REDACTED SSN]",
        "credit_card": "[REDACTED CARD]",
        "email": "[REDACTED EMAIL]",
        "phone_us": "[REDACTED PHONE]",
        "password": "password: [REDACTED]",
        "api_key": "key: [REDACTED]",
    }
    for pii_type, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(replacements.get(pii_type, "[REDACTED]"), redacted)
    return redacted


def validate_input(text: str, allow_pii: bool = False) -> GuardrailResult:
    """
    Full input validation pipeline.

    Args:
        text:       Input text to validate.
        allow_pii:  If True, sanitize PII instead of blocking.

    Returns:
        GuardrailResult with decision and details.
    """
    issues = []

    # Check prompt injection — always block
    if detect_injection(text):
        logger.warning(f"InputValidator: prompt injection detected in: {text[:100]!r}")
        return GuardrailResult(
            decision=GuardrailDecision.BLOCK,
            reason="Prompt injection attempt detected",
            detected_issues=["prompt_injection"],
        )

    # Check harmful content — always block
    harmful = detect_harmful_content(text)
    if harmful:
        logger.warning(f"InputValidator: harmful content detected: {harmful}")
        return GuardrailResult(
            decision=GuardrailDecision.BLOCK,
            reason="Harmful content detected",
            detected_issues=harmful,
        )

    # Check PII
    pii_types = detect_pii(text)
    if pii_types:
        if allow_pii:
            sanitized = redact_pii(text)
            logger.info(f"InputValidator: PII sanitized: {pii_types}")
            return GuardrailResult(
                decision=GuardrailDecision.SANITIZE,
                reason="PII detected and redacted",
                detected_issues=pii_types,
                sanitized_content=sanitized,
            )
        else:
            logger.warning(f"InputValidator: PII detected: {pii_types}")
            return GuardrailResult(
                decision=GuardrailDecision.BLOCK,
                reason="PII detected",
                detected_issues=pii_types,
            )

    return GuardrailResult(decision=GuardrailDecision.ALLOW)
