from langchain_openai import OpenAIEmbeddings

class EmbeddingFactory:
    @staticmethod
    def get_embeddings(provider: str = "openai", model_name: str = "text-embedding-3-small"):
        if provider == "openai":
            return OpenAIEmbeddings(model=model_name)
        # Add HuggingFace, Cohere, etc.
        raise ValueError(f"Provider {provider} not supported.")
