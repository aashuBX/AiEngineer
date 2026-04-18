import re

class MetadataExtractor:
    @staticmethod
    def extract_basics(text: str) -> dict:
        metadata = {}
        # Naive extraction logic
        if re.search(r"Title: (.*)", text):
            metadata["title"] = re.search(r"Title: (.*)", text).group(1)
        return metadata
