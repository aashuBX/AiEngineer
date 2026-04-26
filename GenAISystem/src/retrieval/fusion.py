"""
Reciprocal Rank Fusion — merges and deduplicates ranked document lists.

RRF score formula: sum(1 / (k + rank)) for each list.

Works with LangChain Document objects. Attaches `rrf_score` to doc.metadata
so downstream rerankers can use it as a signal.
"""

import hashlib
from langchain_core.documents import Document


class ReciprocalRankFusion:
    """
    Merge N ranked document lists into a single deduplicated list
    using the Reciprocal Rank Fusion algorithm.
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF constant (60 is standard). Higher k reduces the impact
               of rank differences between lists.
        """
        self.k = k

    def _doc_key(self, doc: Document) -> str:
        """Stable deduplication key based on content hash."""
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    def fuse(self, *result_lists: list[Document]) -> list[Document]:
        """
        Merge multiple ranked document lists using RRF.

        Args:
            *result_lists: Any number of ranked lists of Document objects.

        Returns:
            A single deduplicated list sorted by descending RRF score.
            Each document has `doc.metadata["rrf_score"]` set.
        """
        fused: dict[str, dict] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                key = self._doc_key(doc)
                rrf_score = 1.0 / (rank + self.k)

                if key not in fused:
                    fused[key] = {"doc": doc, "rrf_score": 0.0}
                fused[key]["rrf_score"] += rrf_score

        # Attach RRF scores to metadata and sort descending
        docs_sorted = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

        result = []
        for item in docs_sorted:
            doc = item["doc"]
            doc.metadata["rrf_score"] = round(item["rrf_score"], 6)
            result.append(doc)

        return result
