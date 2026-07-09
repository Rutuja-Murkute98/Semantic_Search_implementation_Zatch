"""
hybrid.py — merges keyword and semantic result lists via Reciprocal Rank
Fusion (RRF). See plan §6 (Phase 5).
"""


def rrf_merge(keyword_results: list, semantic_results: list, k: int = 60) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion:

        score(doc) = sum(1 / (k + rank_in_each_list))

    RRF only uses each result's *rank* within its own list, so keyword
    scores and vector similarity scores never need to be compared directly
    (they're on incompatible scales). Results are deduped by "id".

    Adds a `matchType` field to each result: "keyword", "semantic", or
    "hybrid" (found by both legs).
    """
    scores = {}
    merged = {}

    for rank, item in enumerate(keyword_results, start=1):
        _id = item["id"]
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
        if _id not in merged:
            merged[_id] = dict(item)
            merged[_id]["matchType"] = "keyword"

    for rank, item in enumerate(semantic_results, start=1):
        _id = item["id"]
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank)
        if _id not in merged:
            merged[_id] = dict(item)
            merged[_id]["matchType"] = "semantic"
        else:
            merged[_id]["matchType"] = "hybrid"
            if "vectorScore" in item:
                merged[_id]["vectorScore"] = item["vectorScore"]

    return sorted(merged.values(), key=lambda d: scores[d["id"]], reverse=True)
