class StreamingResponder:
    async def stream_generate(self, llm, query: str, context: list):
        prompt = f"Context: {context}\nQuery: {query}"
        async for chunk in llm.astream(prompt):
            yield chunk
