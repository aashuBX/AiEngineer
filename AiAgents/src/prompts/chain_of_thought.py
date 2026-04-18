"""
Chain-of-Thought, Self-Consistency, and Tree-of-Thought reasoning wrappers.
"""

import asyncio
from collections import Counter
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Chain-of-Thought ───────────────────────────────────────────────────────────

COT_SUFFIX = (
    "\n\nThink step by step before giving your final answer. "
    "Show your reasoning process clearly."
)

ZERO_SHOT_COT_SUFFIX = "\n\nLet's think step by step:"


def build_cot_prompt(question: str, few_shot: bool = False) -> str:
    """Append Chain-of-Thought instruction to a question."""
    suffix = COT_SUFFIX if few_shot else ZERO_SHOT_COT_SUFFIX
    return question + suffix


async def chain_of_thought(
    question: str,
    llm: BaseChatModel,
    system_prompt: str = "You are a careful, step-by-step reasoning assistant.",
) -> str:
    """
    Run Chain-of-Thought reasoning on a question.

    Returns:
        The LLM's reasoned response string.
    """
    logger.debug(f"CoT: reasoning on question ({len(question)} chars)")
    cot_question = build_cot_prompt(question)
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=cot_question),
    ])
    return response.content


# ── Self-Consistency ───────────────────────────────────────────────────────────

async def self_consistency(
    question: str,
    llm: BaseChatModel,
    n_paths: int = 5,
    temperature: float = 0.7,
    system_prompt: str = "You are a careful reasoning assistant.",
) -> dict[str, Any]:
    """
    Self-Consistency: Generate N diverse reasoning paths, then majority vote.

    Args:
        question:      The question to reason about.
        llm:           Base LLM (will be re-invoked N times).
        n_paths:       Number of parallel reasoning paths.
        temperature:   Sampling temperature for diversity.
        system_prompt: System instruction.

    Returns:
        Dict with 'final_answer', 'confidence', 'all_answers', 'reasoning_paths'.
    """
    logger.info(f"Self-Consistency: generating {n_paths} paths")

    # Clone LLM with higher temperature for diversity
    diverse_llm = llm.bind(temperature=temperature)
    cot_question = build_cot_prompt(question)

    async def _single_path() -> str:
        resp = await diverse_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=cot_question),
        ])
        return resp.content

    paths = await asyncio.gather(*[_single_path() for _ in range(n_paths)])

    # Extract final answers (last sentence or line)
    def extract_final(text: str) -> str:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else text[:100]

    answers = [extract_final(p) for p in paths]
    counter = Counter(answers)
    most_common, count = counter.most_common(1)[0]
    confidence = count / n_paths

    logger.info(f"Self-Consistency: consensus={most_common!r}, confidence={confidence:.2f}")

    return {
        "final_answer": most_common,
        "confidence": confidence,
        "all_answers": answers,
        "reasoning_paths": list(paths),
    }


# ── Tree-of-Thought ────────────────────────────────────────────────────────────

async def tree_of_thought(
    question: str,
    llm: BaseChatModel,
    n_branches: int = 3,
    depth: int = 2,
    system_prompt: str = "You are a systematic problem solver.",
) -> dict[str, Any]:
    """
    Tree-of-Thought: Branch → evaluate → prune → continue best paths.

    Args:
        question:    The problem to solve.
        llm:         LLM instance.
        n_branches:  Number of thought branches per node.
        depth:       Number of reasoning steps.
        system_prompt: System instruction.

    Returns:
        Dict with 'best_path', 'final_answer', 'all_thoughts'.
    """
    logger.info(f"ToT: exploring {n_branches} branches × {depth} depth")

    async def _generate_thoughts(context: str, step: int) -> list[str]:
        """Generate N candidate next thoughts."""
        prompt = (
            f"Problem: {question}\n\nThinking so far:\n{context}\n\n"
            f"Generate {n_branches} different possible next reasoning steps "
            f"(Step {step}). Number them 1-{n_branches}."
        )
        resp = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])
        raw = resp.content
        thoughts = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                thoughts.append(line.split(". ", 1)[1].strip())
        return thoughts[:n_branches] if thoughts else [raw]

    async def _evaluate_thought(context: str, thought: str) -> float:
        """Score a thought path (0-1)."""
        prompt = (
            f"Problem: {question}\n\nThought path:\n{context}\nNext: {thought}\n\n"
            "Rate the promise of this reasoning path on a scale of 0.0 to 1.0. "
            "Respond with ONLY a float."
        )
        try:
            resp = await llm.ainvoke([
                SystemMessage(content="You are a reasoning evaluator."),
                HumanMessage(content=prompt),
            ])
            return float(resp.content.strip())
        except Exception:
            return 0.5

    # BFS-style exploration
    paths = [("", 1.0)]  # (context, score)
    all_thoughts: list[str] = []

    for step in range(1, depth + 1):
        new_paths = []
        for context, score in paths:
            thoughts = await _generate_thoughts(context, step)
            all_thoughts.extend(thoughts)

            # Evaluate and keep top branch
            scores = await asyncio.gather(*[_evaluate_thought(context, t) for t in thoughts])
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_thought = thoughts[best_idx]
            new_context = f"{context}\nStep {step}: {best_thought}".strip()
            new_paths.append((new_context, scores[best_idx]))

        # Prune to best path
        paths = sorted(new_paths, key=lambda x: x[1], reverse=True)[:1]

    best_context, best_score = paths[0]

    # Generate final answer from best path
    final_resp = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Problem: {question}\n\nReasoning:\n{best_context}\n\nFinal Answer:"),
    ])
    final_answer = final_resp.content

    return {
        "best_path": best_context,
        "final_answer": final_answer,
        "path_score": best_score,
        "all_thoughts": all_thoughts,
    }
