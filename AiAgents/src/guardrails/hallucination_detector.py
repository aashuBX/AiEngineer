"""
Hallucination Detector — tiered post-generation cascade for AiAgents.

Three detection tiers, run in ascending cost order with early exit:

  Tier 1 — Semantic Similarity (fast, ~10ms)
      Computes cosine similarity between the LLM answer and the retrieved
      context using sentence-transformer embeddings. Catches paraphrased
      hallucinations that exact word-overlap misses.

      Fast-pass:  similarity ≥ pass_threshold  → FAITHFUL (skip Tier 2/3)
      Fast-fail:  similarity <  fail_threshold  → HALLUCINATED (skip Tier 2/3)
      Borderline: continue to Tier 2

  Tier 2 — LLM-as-a-Judge (medium, ~200ms)
      Sends the context + answer to an LLM judge with a structured faithfulness
      prompt. Returns a JSON verdict: faithful / partially_faithful / hallucinated,
      plus a list of unsupported claims. This is the technique used by Ragas,
      TruLens, and LangSmith evaluators in production RAG systems.

  Tier 3 — Claim Decomposition & Verification (thorough, ~500ms)
      Decomposes the answer into atomic factual claims, then verifies each claim
      individually against the context using semantic similarity. Pinpoints
      exactly which sentences are fabricated. Enabled in STRICT_SAFETY_CONFIG.

Public API:
  run_hallucination_cascade(answer, context, config) -> HallucinationResult
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from src.models.schemas import HallucinationResult, HallucinationVerdict
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.guardrails.safety_config import SafetyConfig

logger = get_logger(__name__)

# ── Faithfulness judge prompt ──────────────────────────────────────────────────
_FAITHFULNESS_JUDGE_PROMPT = """\
You are a strict Faithfulness Evaluator for an AI system.

You will receive:
  CONTEXT: The retrieved source documents used to generate the answer.
  ANSWER:  The AI-generated answer that needs to be verified.

Your task: determine whether EVERY factual claim in the ANSWER is directly
supported by the CONTEXT. Be conservative — if a claim goes even slightly
beyond what the CONTEXT says, mark it as unsupported.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "verdict": "faithful" | "partially_faithful" | "hallucinated",
  "confidence": <float 0.0-1.0>,
  "unsupported_claims": ["exact quote of each claim NOT supported by context"],
  "reasoning": "one sentence explanation"
}}

Rules:
- "faithful"           → ALL claims are supported by context
- "partially_faithful" → SOME claims are unsupported (list them)
- "hallucinated"       → MOST claims are fabricated / not in context
- If CONTEXT is empty, always return "hallucinated"

CONTEXT:
{context}

ANSWER:
{answer}
"""

# ── Claim decomposition prompt ─────────────────────────────────────────────────
_CLAIM_DECOMPOSE_PROMPT = """\
Break the following text into a list of atomic, self-contained factual claims.
Each claim should be a single sentence that can be independently verified.
Ignore opinions, caveats, and uncertain statements (they are not factual claims).

Respond ONLY with a JSON array of strings:
["claim 1", "claim 2", ...]

TEXT:
{text}
"""


# ── Tier 1: Semantic Similarity ────────────────────────────────────────────────

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Pure-Python cosine similarity (avoids mandatory numpy import)."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_embedding_model():
    """Lazy-load sentence-transformers (optional dependency)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Tier 1 semantic similarity will use TF-IDF fallback. "
            "Run: pip install sentence-transformers"
        )
        return None


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """
    Simple TF-IDF cosine similarity fallback when sentence-transformers
    is not installed. Better than word-overlap — handles common words correctly.
    """
    import math

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w{3,}\b", text.lower())

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    vocab = list(set(tokens_a) | set(tokens_b))
    all_docs = [tokens_a, tokens_b]

    def tf(tokens: list[str], word: str) -> float:
        return tokens.count(word) / len(tokens) if tokens else 0.0

    def idf(docs: list[list[str]], word: str) -> float:
        n_docs = sum(1 for d in docs if word in d)
        return math.log((len(docs) + 1) / (n_docs + 1)) + 1.0

    def tfidf_vec(tokens: list[str]) -> list[float]:
        return [tf(tokens, w) * idf(all_docs, w) for w in vocab]

    vec_a = tfidf_vec(tokens_a)
    vec_b = tfidf_vec(tokens_b)
    return _cosine_similarity(vec_a, vec_b)


def check_semantic_grounding(answer: str, context: str) -> float:
    """
    Tier 1: Compute semantic similarity between the answer and context.

    Returns a float in [0, 1]. Higher → answer is more grounded in context.
    Uses sentence-transformers if available, otherwise falls back to TF-IDF.
    """
    if not context.strip():
        return 0.0

    model = _get_embedding_model()
    if model is not None:
        try:
            embeddings = model.encode([answer, context])
            similarity = _cosine_similarity(
                embeddings[0].tolist(), embeddings[1].tolist()
            )
            logger.debug(f"HallucinationDetector[Tier1] embedding similarity: {similarity:.3f}")
            return float(similarity)
        except Exception as e:
            logger.warning(f"HallucinationDetector[Tier1] embedding failed, using TF-IDF fallback: {e}")

    similarity = _tfidf_similarity(answer, context)
    logger.debug(f"HallucinationDetector[Tier1] TF-IDF similarity: {similarity:.3f}")
    return similarity


# ── Tier 2: LLM-as-a-Judge ────────────────────────────────────────────────────

async def check_faithfulness_llm(
    answer: str,
    context: str,
    confidence_threshold: float = 0.65,
) -> dict:
    """
    Tier 2: Ask an LLM judge whether the answer is faithful to the context.

    Returns a dict with keys:
      verdict: "faithful" | "partially_faithful" | "hallucinated"
      confidence: float
      unsupported_claims: list[str]
      reasoning: str
    """
    from src.config.llm_providers import get_default_llm

    # Truncate very long contexts to avoid token limits (keep first 3000 chars)
    context_snippet = context[:3000] if len(context) > 3000 else context
    prompt = _FAITHFULNESS_JUDGE_PROMPT.format(
        context=context_snippet, answer=answer
    )

    llm = get_default_llm()
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        response = await llm.ainvoke([
            SystemMessage(content="You are a faithfulness evaluator. Respond only with valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        verdict = data.get("verdict", "hallucinated")
        confidence = float(data.get("confidence", 0.5))
        unsupported = data.get("unsupported_claims", [])
        reasoning = data.get("reasoning", "")

        logger.info(
            f"HallucinationDetector[Tier2] verdict={verdict!r} "
            f"confidence={confidence:.2f} unsupported={len(unsupported)} claims"
        )
        return {
            "verdict": verdict,
            "confidence": confidence,
            "unsupported_claims": unsupported,
            "reasoning": reasoning,
        }

    except Exception as e:
        logger.error(f"HallucinationDetector[Tier2] LLM judge failed: {e}")
        # Safe default: treat as partially faithful (don't block on detector failure)
        return {
            "verdict": "partially_faithful",
            "confidence": 0.5,
            "unsupported_claims": [],
            "reasoning": f"Judge error: {e}",
        }


# ── Tier 3: Claim Decomposition & Verification ────────────────────────────────

async def decompose_and_verify_claims(answer: str, context: str) -> dict:
    """
    Tier 3: Decompose the answer into atomic factual claims, then verify each
    against the context using semantic similarity.

    Returns a dict with keys:
      total_claims: int
      supported_claims: int
      unsupported_claims: list[str]
      support_ratio: float  (supported / total, or 1.0 if no claims)
    """
    from src.config.llm_providers import get_default_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_default_llm()

    # Step 1: Decompose answer into atomic claims
    try:
        decompose_prompt = _CLAIM_DECOMPOSE_PROMPT.format(text=answer)
        response = await llm.ainvoke([
            SystemMessage(content="Extract factual claims. Respond only with a JSON array of strings."),
            HumanMessage(content=decompose_prompt),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        claims: list[str] = json.loads(raw)
        if not isinstance(claims, list):
            claims = []
    except Exception as e:
        logger.error(f"HallucinationDetector[Tier3] claim decomposition failed: {e}")
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "support_ratio": 1.0,
        }

    if not claims:
        logger.debug("HallucinationDetector[Tier3] no factual claims found in answer")
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": [],
            "support_ratio": 1.0,
        }

    # Step 2: Verify each claim against context using semantic similarity
    # Threshold: claim is "supported" if similarity > 0.4
    CLAIM_SUPPORT_THRESHOLD = 0.4
    supported = []
    unsupported = []

    for claim in claims:
        sim = check_semantic_grounding(claim, context)
        if sim >= CLAIM_SUPPORT_THRESHOLD:
            supported.append(claim)
        else:
            unsupported.append(claim)
            logger.debug(f"HallucinationDetector[Tier3] unsupported claim (sim={sim:.2f}): {claim!r}")

    support_ratio = len(supported) / len(claims)
    logger.info(
        f"HallucinationDetector[Tier3] claims: {len(supported)}/{len(claims)} supported "
        f"({support_ratio:.0%})"
    )

    return {
        "total_claims": len(claims),
        "supported_claims": len(supported),
        "unsupported_claims": unsupported,
        "support_ratio": support_ratio,
    }


# ── Orchestrator: Tiered Cascade ───────────────────────────────────────────────

async def run_hallucination_cascade(
    answer: str,
    context: str,
    config: "SafetyConfig | None" = None,
) -> HallucinationResult:
    """
    Run the tiered hallucination detection cascade.

    Tiers are run in order (fast → slow) with early-exit on a definitive result.
    The config controls which tiers are enabled and the decision thresholds.

    Args:
        answer:  The LLM-generated response text to evaluate.
        context: The retrieved context used for generation (from RAG pipeline).
        config:  SafetyConfig instance. Defaults to DEFAULT_SAFETY_CONFIG.

    Returns:
        HallucinationResult with verdict, risk_score, and audit details.
    """
    from src.guardrails.safety_config import DEFAULT_SAFETY_CONFIG
    cfg = config or DEFAULT_SAFETY_CONFIG

    if not cfg.hallucination_detection_enabled:
        logger.debug("HallucinationDetector: detection disabled by config — skipping")
        return HallucinationResult(
            verdict=HallucinationVerdict.FAITHFUL,
            risk_score=0.0,
            detection_tiers_run=[],
        )

    # Bail out early if there is no context to ground against
    if not context or not context.strip():
        logger.info("HallucinationDetector: no context provided — skipping grounding check")
        return HallucinationResult(
            verdict=HallucinationVerdict.FAITHFUL,
            risk_score=0.0,
            detection_tiers_run=[],
        )

    if not answer or not answer.strip():
        return HallucinationResult(
            verdict=HallucinationVerdict.FAITHFUL,
            risk_score=0.0,
            detection_tiers_run=[],
        )

    enabled_tiers: list[str] = cfg.hallucination_tiers
    tiers_run: list[str] = []

    # Accumulated result fields
    semantic_sim: float | None = None
    llm_verdict: str | None = None
    llm_confidence: float | None = None
    unsupported_claims: list[str] = []
    total_claims: int | None = None
    supported_claims_count: int | None = None
    risk_score = 0.0

    # ── Tier 1: Semantic Similarity ───────────────────────────────────────
    if "semantic" in enabled_tiers:
        tiers_run.append("semantic")
        semantic_sim = check_semantic_grounding(answer, context)

        fail_thresh = cfg.semantic_similarity_fail_threshold
        pass_thresh = cfg.semantic_similarity_pass_threshold

        if semantic_sim < fail_thresh:
            # Very low grounding → immediate FAIL, skip expensive tiers
            risk_score = 1.0 - semantic_sim
            logger.warning(
                f"HallucinationDetector[Tier1] FAST-FAIL: similarity={semantic_sim:.3f} "
                f"< threshold={fail_thresh}"
            )
            return HallucinationResult(
                verdict=HallucinationVerdict.HALLUCINATED,
                risk_score=round(risk_score, 3),
                semantic_similarity=semantic_sim,
                detection_tiers_run=tiers_run,
                retry_suggested=True,
            )

        if semantic_sim >= pass_thresh and "llm_judge" in enabled_tiers:
            # High grounding → fast pass, skip LLM judge to save cost
            logger.info(
                f"HallucinationDetector[Tier1] FAST-PASS: similarity={semantic_sim:.3f} "
                f">= threshold={pass_thresh} — skipping Tier 2"
            )
            return HallucinationResult(
                verdict=HallucinationVerdict.FAITHFUL,
                risk_score=round(1.0 - semantic_sim, 3),
                semantic_similarity=semantic_sim,
                detection_tiers_run=tiers_run,
                retry_suggested=False,
            )

        # Borderline — accumulate partial risk
        risk_score += (1.0 - semantic_sim) * 0.4
        logger.info(
            f"HallucinationDetector[Tier1] borderline: similarity={semantic_sim:.3f}, "
            f"continuing to Tier 2"
        )

    # ── Tier 2: LLM-as-a-Judge ────────────────────────────────────────────
    if "llm_judge" in enabled_tiers:
        tiers_run.append("llm_judge")
        judge_result = await check_faithfulness_llm(
            answer, context, cfg.faithfulness_confidence_threshold
        )
        llm_verdict = judge_result["verdict"]
        llm_confidence = judge_result["confidence"]
        unsupported_claims = judge_result.get("unsupported_claims", [])

        if llm_verdict == "hallucinated":
            risk_score = min(risk_score + 0.6, 1.0)
        elif llm_verdict == "partially_faithful":
            risk_score = min(risk_score + 0.3, 1.0)
        else:  # faithful
            risk_score = max(risk_score - 0.2, 0.0)

        # If judge is confident and says faithful, no need for Tier 3
        if llm_verdict == "faithful" and llm_confidence >= cfg.faithfulness_confidence_threshold:
            logger.info("HallucinationDetector[Tier2] confident FAITHFUL — skipping Tier 3")
            return HallucinationResult(
                verdict=HallucinationVerdict.FAITHFUL,
                risk_score=round(risk_score, 3),
                semantic_similarity=semantic_sim,
                llm_judge_confidence=llm_confidence,
                unsupported_claims=unsupported_claims,
                detection_tiers_run=tiers_run,
                retry_suggested=False,
            )

        # If judge is confident and says hallucinated, Tier 3 can confirm
        if llm_verdict == "hallucinated" and "claim_decomp" not in enabled_tiers:
            return HallucinationResult(
                verdict=HallucinationVerdict.HALLUCINATED,
                risk_score=round(risk_score, 3),
                semantic_similarity=semantic_sim,
                llm_judge_confidence=llm_confidence,
                unsupported_claims=unsupported_claims,
                detection_tiers_run=tiers_run,
                retry_suggested=True,
            )

    # ── Tier 3: Claim Decomposition & Verification ────────────────────────
    if "claim_decomp" in enabled_tiers:
        tiers_run.append("claim_decomp")
        claim_result = await decompose_and_verify_claims(answer, context)
        total_claims = claim_result["total_claims"]
        supported_claims_count = claim_result["supported_claims"]
        support_ratio = claim_result["support_ratio"]

        # Merge with existing unsupported claims list
        for c in claim_result["unsupported_claims"]:
            if c not in unsupported_claims:
                unsupported_claims.append(c)

        # Adjust risk score based on claim support ratio
        if total_claims and total_claims > 0:
            risk_score = min(risk_score + (1.0 - support_ratio) * 0.4, 1.0)

    # ── Final verdict based on accumulated risk score ─────────────────────
    if risk_score >= 0.6:
        final_verdict = HallucinationVerdict.HALLUCINATED
        retry = True
    elif risk_score >= 0.3:
        final_verdict = HallucinationVerdict.PARTIALLY_FAITHFUL
        retry = True
    else:
        final_verdict = HallucinationVerdict.FAITHFUL
        retry = False

    logger.info(
        f"HallucinationDetector cascade complete: verdict={final_verdict.value} "
        f"risk={risk_score:.3f} tiers={tiers_run}"
    )

    return HallucinationResult(
        verdict=final_verdict,
        risk_score=round(risk_score, 3),
        semantic_similarity=semantic_sim,
        llm_judge_confidence=llm_confidence,
        unsupported_claims=unsupported_claims,
        total_claims=total_claims,
        supported_claims=supported_claims_count,
        detection_tiers_run=tiers_run,
        retry_suggested=retry,
    )
