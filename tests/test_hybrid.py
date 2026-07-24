"""
test_hybrid.py — sanity tests for RRF merge and relevance score ordering.

Run with pytest, or directly:
    python tests/test_hybrid.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.hybrid import rrf_merge


def _is_monotonic_non_increasing(scores):
    return all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_relevance_score_is_monotonic_for_mixed_matches():
    # Regression test: relevanceScore must always match display order --
    # it's the actual combined sort key, not a raw per-leg score (which
    # can disagree with the true order, e.g. a "hybrid" match ranking
    # above a "semantic" one with a higher raw vectorScore).
    keyword = [
        {"id": "a", "name": "Track Pants"},
        {"id": "c", "name": "Soft Cotton Saree"},
    ]
    semantic = [
        {"id": "b", "name": "Cotton Short Kurtis", "vectorScore": 0.90},
        {"id": "a", "name": "Track Pants", "vectorScore": 0.85},
        {"id": "d", "name": "Pure mul cotton saree", "vectorScore": 0.80},
        {"id": "c", "name": "Soft Cotton Saree", "vectorScore": 0.70},
    ]
    merged = rrf_merge(keyword, semantic)
    scores = [item["relevanceScore"] for item in merged]
    assert _is_monotonic_non_increasing(scores)


def test_relevance_score_top_result_is_one():
    merged = rrf_merge([{"id": "x", "name": "Only"}], [])
    assert merged[0]["relevanceScore"] == 1.0


def test_relevance_score_empty_lists():
    assert rrf_merge([], []) == []


def test_hybrid_match_type_when_found_by_both_legs():
    keyword = [{"id": "a", "name": "Item"}]
    semantic = [{"id": "a", "name": "Item", "vectorScore": 0.9}]
    merged = rrf_merge(keyword, semantic)
    assert merged[0]["matchType"] == "hybrid"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll tests passed.")
