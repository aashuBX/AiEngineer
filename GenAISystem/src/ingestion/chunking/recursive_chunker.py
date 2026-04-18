from langchain.text_splitter import RecursiveCharacterTextSplitter

class AdvancedRecursiveChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str):
        return self.splitter.split_text(text)
    
    def split_documents(self, documents: list):
        return self.splitter.split_documents(documents)
