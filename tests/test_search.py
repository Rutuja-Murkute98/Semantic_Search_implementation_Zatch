"""
test_search.py — sanity tests for query-parsing logic in app/search.py.

Run with pytest, or directly:
    python tests/test_search.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.search import pick_main_term, extract_colors, search_tokens


def test_pick_main_term_skips_leading_color():
    # Regression test: "red saree" must not pick "red" as the main term
    # just because it comes first -- color is a modifier role, "saree" is
    # the actual subject being searched.
    tokens = search_tokens("red saree")
    colors = extract_colors("red saree")
    assert pick_main_term(tokens, colors) == "saree"


def test_pick_main_term_falls_back_to_first_token_if_all_are_colors():
    tokens = ["red", "blue"]
    colors = {"red", "blue"}
    assert pick_main_term(tokens, colors) == "red"


def test_pick_main_term_no_colors_uses_first_token():
    tokens = search_tokens("formal shirt")
    colors = extract_colors("formal shirt")
    assert pick_main_term(tokens, colors) == "formal"


def test_pick_main_term_empty_tokens():
    assert pick_main_term([], set()) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll tests passed.")
