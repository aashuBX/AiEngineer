"""
Structured Output Enforcement — JSON mode, Pydantic parsing, schema-constrained generation.
"""

from typing import Any, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_structured_llm(llm: BaseChatModel, schema: Type[BaseModel]) -> Any:
    """
    Bind a Pydantic schema to an LLM using .with_structured_output().
    The LLM will return a validated Pydantic model instance.

    Args:
        llm:    Base LLM.
        schema: Pydantic model class defining the output schema.

    Returns:
        An LLM runnable that returns schema instances.
    """
    logger.debug(f"Binding structured output schema: {schema.__name__}")
    return llm.with_structured_output(schema)


def get_json_output_parser() -> JsonOutputParser:
    """Return a JSON output parser for free-form JSON responses."""
    return JsonOutputParser()


async def invoke_with_structured_output(
    llm: BaseChatModel,
    schema: Type[BaseModel],
    prompt: ChatPromptTemplate,
    input_vars: dict[str, Any],
    max_retries: int = 3,
) -> BaseModel:
    """
    Invoke an LLM with structured output and retry on validation failures.

    Args:
        llm:         Base LLM.
        schema:      Target Pydantic model.
        prompt:      ChatPromptTemplate to format.
        input_vars:  Variables to fill the prompt.
        max_retries: Number of retry attempts on parse failure.

    Returns:
        A validated Pydantic model instance.

    Raises:
        ValueError: If all retries fail.
    """
    structured_llm = get_structured_llm(llm, schema)
    chain = prompt | structured_llm

    for attempt in range(1, max_retries + 1):
        try:
            result = await chain.ainvoke(input_vars)
            logger.debug(f"Structured output succeeded on attempt {attempt}")
            return result
        except (ValidationError, Exception) as e:
            logger.warning(f"Structured output attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise ValueError(
                    f"Failed to get valid structured output after {max_retries} attempts"
                ) from e

    raise ValueError("Structured output extraction failed")


def extract_json_from_text(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a text response that may contain markdown fences.

    Args:
        text: Raw LLM output string.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON is found.
    """
    import json
    import re

    # Try to extract from markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in text: {text[:200]}...")


def parse_with_retry(text: str, schema: Type[BaseModel]) -> BaseModel:
    """
    Parse a JSON string into a Pydantic model with error reporting.

    Args:
        text:   Raw JSON string.
        schema: Target Pydantic model class.

    Returns:
        Validated model instance.
    """
    try:
        data = extract_json_from_text(text)
        return schema(**data)
    except (ValidationError, ValueError) as e:
        raise ValueError(f"Failed to parse {schema.__name__}: {e}") from e
