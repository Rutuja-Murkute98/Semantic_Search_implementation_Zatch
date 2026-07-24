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


def test_pick_main_term_no_modifiers_uses_first_token():
    # Neither word is a colour/material/pattern modifier, so the first
    # token wins by default.
    tokens = search_tokens("necklace bag")
    colors = extract_colors("necklace bag")
    assert pick_main_term(tokens, colors) == "necklace"


def test_pick_main_term_skips_material():
    # Regression test: "cotton pants" must not pick "cotton" as the main
    # term -- material is a modifier role, "pants" is the actual subject.
    tokens = search_tokens("cotton pants")
    colors = extract_colors("cotton pants")
    assert pick_main_term(tokens, colors) == "pants"


def test_pick_main_term_skips_pattern_and_color():
    # Regression test: colours AND pattern words ("striped") must both be
    # skipped so the product noun ("kurta") is chosen, not the last
    # modifier standing.
    tokens = search_tokens("black and white striped kurta")
    colors = extract_colors("black and white striped kurta")
    assert pick_main_term(tokens, colors) == "kurta"


def test_pick_main_term_empty_tokens():
    assert pick_main_term([], set()) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll tests passed.")
