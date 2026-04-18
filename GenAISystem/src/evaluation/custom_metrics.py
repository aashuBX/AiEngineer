class CustomMetrics:
    @staticmethod
    def check_hallucination(answer: str, context: str, llm) -> float:
        # Ask LLM-as-judge if answer is supported by context
        return 0.0
