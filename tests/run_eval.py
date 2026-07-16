"""
run_eval.py — runs tests/golden_queries.json through do_search() and reports
a hit-rate, so you can tune RRF k / similarity cutoff / numCandidates and
see the effect (plan §6, Phase 6).

Usage:
    python tests/run_eval.py

Cases with an empty "expectedIds" list are skipped (nothing to check yet) —
fill them in with real product/seller _id strings from your catalog.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.search import do_search

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_queries.json")


def load_golden(path=GOLDEN_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["queries"]


def evaluate():
    golden = load_golden()
    scored = skipped = hits = 0

    for case in golden:
        expected = set(case.get("expectedIds") or [])
        if not expected:
            skipped += 1
            print(f"SKIP  {case['query']!r} (no expectedIds set)")
            continue

        result = do_search(case["query"])
        found_ids = {p["id"] for p in result.get("products", [])} | \
                    {s["id"] for s in result.get("sellers", [])}

        hit = bool(found_ids & expected)
        hits += hit
        scored += 1
        mark = "PASS" if hit else "FAIL"
        print(f"{mark}  {case['query']!r} (mode={result.get('searchMode')})")

    print()
    if scored:
        print(f"Hit rate: {hits}/{scored} ({hits / scored:.0%})")
    print(f"Skipped (no expectedIds): {skipped}")


if __name__ == "__main__":
    evaluate()
