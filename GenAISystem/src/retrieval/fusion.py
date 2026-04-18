class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, *result_lists):
        # RRF implementation
        fused_scores = {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", str(doc))
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"doc": doc, "score": 0.0}
                fused_scores[doc_id]["score"] += 1.0 / (rank + self.k)
                
        sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results]
