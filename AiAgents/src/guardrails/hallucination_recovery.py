"""
Hallucination Recovery — conversation continuation strategies after detection.

When the hallucination detector flags a response, this module provides three
recovery strategies tried in order:

  Strategy 1 — Constrained Re-generation  (attempt == 0)
      Re-invoke the LLM with the unsupported claims explicitly listed in the
      prompt, instructing it to answer ONLY from the provided context. This
      catches most cases where the model simply drifted beyond the context.

  Strategy 2 — Extractive Fallback  (attempt > 0, context available)
      Rather than generating, extract the most semantically relevant sentences
      directly from the retrieved context and return them as bullet points with
      a transparency disclaimer. Zero hallucination risk — pure extraction.

  Strategy 3 — Transparent Degradation  (no context, or all strategies fail)
      Return a calibrated uncertainty message that is honest with the user
      about the system's limitations and offers a path forward (clarify the
      question or escalate to a human agent).

Key design principle: the user always receives a response — never a dead-end
error. Transparency is always preferred over silence or confident fabrication.

Public API:
  recover_from_hallucination(
      original_answer, context, user_query,
      hallucination_result, attempt, max_attempts
  ) -> RecoveryResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.models.schemas import HallucinationResult

logger = get_logger(__name__)


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class RecoveryResult:
    """
    The output of the recovery process.

    Attributes:
        answer:          The final answer to return to the user.
        strategy_used:   Which strategy produced this answer.
        was_recovered:   True if re-generation or extraction was used.
        recovery_reason: Human-readable explanation for transparency.
        confidence:      Estimated confidence in the recovered answer.
    """
    answer: str
    strategy_used: str
    was_recovered: bool = False
    recovery_reason: str = ""
    confidence: float = 1.0


# ── Strategy 1: Constrained Re-generation ─────────────────────────────────────

_CONSTRAINED_REGEN_PROMPT = """\
You are answering a user's question. You MUST use ONLY the information in the \
provided CONTEXT below. Do not add any information that is not directly stated \
in the CONTEXT.

{unsupported_block}

CONTEXT (your only allowed source of information):
{context}

USER QUESTION:
{user_query}

Rules:
- If the CONTEXT contains a clear answer, provide it concisely.
- If the CONTEXT does NOT contain enough information to fully answer, say:
  "Based on the available information, I can tell you that [what context says].
   However, I don't have complete information about [specific gap]."
- Never fabricate details. Never guess. Never use outside knowledge.
"""

_UNSUPPORTED_BLOCK_TEMPLATE = """\
IMPORTANT — The following claims were detected as NOT supported by the context \
in a previous response. Do NOT repeat them:
{claims}
"""


async def _constrained_regenerate(
    context: str,
    user_query: str,
    unsupported_claims: list[str],
    system_prompt: str = "",
) -> str:
    """
    Re-generate an answer with an explicit constraint prompt listing the
    claims that were previously hallucinated.
    """
    from src.config.llm_providers import get_default_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    unsupported_block = ""
    if unsupported_claims:
        claims_formatted = "\n".join(f"  - {c}" for c in unsupported_claims)
        unsupported_block = _UNSUPPORTED_BLOCK_TEMPLATE.format(claims=claims_formatted)

    # Truncate context if too long
    context_snippet = context[:4000] if len(context) > 4000 else context

    prompt = _CONSTRAINED_REGEN_PROMPT.format(
        unsupported_block=unsupported_block,
        context=context_snippet,
        user_query=user_query,
    )

    llm = get_default_llm()
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    try:
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"HallucinationRecovery[Strategy1] re-generation failed: {e}")
        return ""


# ── Strategy 2: Extractive Fallback ───────────────────────────────────────────

def _extract_relevant_sentences(context: str, user_query: str, top_n: int = 5) -> list[str]:
    """
    Extract the most relevant sentences from context using TF-IDF-style
    keyword matching. No LLM needed — pure extraction.
    """
    # Split context into sentences
    sentences = re.split(r"(?<=[.!?])\s+", context.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return []

    # Score each sentence by keyword overlap with the user query
    query_words = set(re.findall(r"\b\w{4,}\b", user_query.lower()))

    scored = []
    for sent in sentences:
        sent_words = set(re.findall(r"\b\w{4,}\b", sent.lower()))
        overlap = len(query_words & sent_words)
        scored.append((overlap, sent))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [sent for score, sent in scored[:top_n] if score > 0]

    return top


_EXTRACTIVE_DISCLAIMER = (
    "Based strictly on the available documents, here is what I found:\n\n"
    "{bullet_points}\n\n"
    "*Note: I was unable to synthesize a fully confident answer. "
    "The above are direct excerpts from the knowledge base relevant to your question.*"
)

_EXTRACTIVE_EMPTY_DISCLAIMER = (
    "I wasn't able to find specific information about your question in the available documents. "
    "The knowledge base may not contain this information, or you may want to rephrase your question "
    "to be more specific."
)


def _build_extractive_response(context: str, user_query: str) -> str:
    """
    Build a response from direct context extractions instead of LLM synthesis.
    """
    relevant_sentences = _extract_relevant_sentences(context, user_query, top_n=5)

    if not relevant_sentences:
        return _EXTRACTIVE_EMPTY_DISCLAIMER

    bullet_points = "\n".join(f"• {s}" for s in relevant_sentences)
    return _EXTRACTIVE_DISCLAIMER.format(bullet_points=bullet_points)


# ── Strategy 3: Transparent Degradation ───────────────────────────────────────

_TRANSPARENT_DEGRADATION_MESSAGES = {
    "no_context": (
        "I want to be transparent — I'm not fully confident in the accuracy of my previous response. "
        "I don't have access to specific documents that would let me verify this answer. "
        "Could you help me by:\n"
        "• Rephrasing or narrowing your question\n"
        "• Asking about a specific aspect you need most\n"
        "• Requesting to speak with a human agent for a verified answer"
    ),
    "retry_exhausted": (
        "I apologize, but I'm having difficulty providing a fully accurate answer to your question "
        "from the available information. To ensure I don't give you incorrect information, "
        "I'd recommend:\n"
        "• Speaking with a human agent who can verify the details\n"
        "• Checking the source documentation directly\n"
        "• Trying a more specific version of your question"
    ),
}


# ── Main Recovery Orchestrator ─────────────────────────────────────────────────

async def recover_from_hallucination(
    original_answer: str,
    context: str,
    user_query: str,
    hallucination_result: "HallucinationResult",
    attempt: int = 0,
    max_attempts: int = 1,
    system_prompt: str = "",
) -> RecoveryResult:
    """
    Orchestrate hallucination recovery using the appropriate strategy.

    Args:
        original_answer:      The hallucinated answer that was rejected.
        context:              The retrieved context that was used for generation.
        user_query:           The original user question.
        hallucination_result: HallucinationResult from the detector cascade.
        attempt:              Number of recovery attempts already made (starts at 0).
        max_attempts:         Maximum number of re-generation attempts allowed.
        system_prompt:        Optional system prompt to include in re-generation.

    Returns:
        RecoveryResult with the recovered answer and metadata.
    """
    has_context = bool(context and context.strip())
    unsupported = hallucination_result.unsupported_claims or []
    verdict = hallucination_result.verdict.value

    logger.info(
        f"HallucinationRecovery: starting recovery "
        f"attempt={attempt}/{max_attempts} verdict={verdict!r} "
        f"unsupported_claims={len(unsupported)} has_context={has_context}"
    )

    # ── Strategy 1: Constrained Re-generation ─────────────────────────────
    if attempt < max_attempts and has_context:
        logger.info("HallucinationRecovery: trying Strategy 1 — constrained re-generation")
        regenerated = await _constrained_regenerate(
            context=context,
            user_query=user_query,
            unsupported_claims=unsupported,
            system_prompt=system_prompt,
        )

        if regenerated:
            # Run a quick verification on the regenerated answer
            from src.guardrails.hallucination_detector import check_semantic_grounding
            sim = check_semantic_grounding(regenerated, context)

            if sim >= 0.3:  # Passes basic sanity check
                logger.info(
                    f"HallucinationRecovery[Strategy1] success: "
                    f"re-generated answer similarity={sim:.3f}"
                )
                return RecoveryResult(
                    answer=regenerated,
                    strategy_used="constrained_regeneration",
                    was_recovered=True,
                    recovery_reason=(
                        f"Original response contained {len(unsupported)} unsupported claim(s). "
                        f"Answer was re-generated with strict context constraints."
                    ),
                    confidence=round(sim, 2),
                )
            else:
                logger.warning(
                    f"HallucinationRecovery[Strategy1] re-generated answer still low similarity "
                    f"({sim:.3f}), falling through to Strategy 2"
                )

    # ── Strategy 2: Extractive Fallback ───────────────────────────────────
    if has_context:
        logger.info("HallucinationRecovery: trying Strategy 2 — extractive fallback")
        extracted_response = _build_extractive_response(context, user_query)

        logger.info("HallucinationRecovery[Strategy2] returning extractive response")
        return RecoveryResult(
            answer=extracted_response,
            strategy_used="extractive_fallback",
            was_recovered=True,
            recovery_reason=(
                f"Re-generation did not produce a sufficiently grounded answer. "
                f"Returning direct context excerpts instead."
            ),
            confidence=0.95,  # High confidence — no synthesis involved
        )

    # ── Strategy 3: Transparent Degradation ───────────────────────────────
    logger.info("HallucinationRecovery: falling back to Strategy 3 — transparent degradation")

    if attempt >= max_attempts:
        degradation_message = _TRANSPARENT_DEGRADATION_MESSAGES["retry_exhausted"]
        reason = "Maximum re-generation attempts exhausted."
    else:
        degradation_message = _TRANSPARENT_DEGRADATION_MESSAGES["no_context"]
        reason = "No retrieved context available for grounding check."

    return RecoveryResult(
        answer=degradation_message,
        strategy_used="transparent_degradation",
        was_recovered=True,
        recovery_reason=reason,
        confidence=0.0,
    )
