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
    "hybrid" (found by both legs). Also adds `relevanceScore`, the actual
    combined score each result was sorted by, normalized to the top
    result in this list (=1.0). This matters beyond cosmetics: the old
    per-item `vectorScore` field only ever held the *semantic* leg's raw
    score, which does not necessarily agree with the true (keyword +
    semantic combined) sort order -- a "hybrid" result can rank above a
    "semantic" one with a higher raw vectorScore, because its keyword-side
    rank contributed too. Displaying vectorScore next to results sorted
    by the combined score could then show a lower number ranked above a
    higher one, which reads as a ranking bug even though the order itself
    is correct. relevanceScore is guaranteed monotonically non-increasing
    down the list because it *is* the sort key, just rescaled to be
    readable.
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

    ordered = sorted(merged.values(), key=lambda d: scores[d["id"]], reverse=True)

    top_score = scores[ordered[0]["id"]] if ordered else 0.0
    for item in ordered:
        item["relevanceScore"] = (
            round(scores[item["id"]] / top_score, 4) if top_score else 0.0
        )

    return ordered
