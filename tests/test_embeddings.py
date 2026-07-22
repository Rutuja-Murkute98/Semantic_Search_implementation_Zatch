"""
test_embeddings.py — sanity tests for the text builders in app/embeddings.py.

Run with pytest, or directly:
    python tests/test_embeddings.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import product_to_text, seller_to_text, bit_to_text, text_hash


def test_product_to_text_includes_key_fields():
    product = {
        "name": "Banarasi Silk Saree",
        "category": "Women",
        "subCategory": "Sarees",
        "tags": ["wedding", "festive", "silk"],
        "searchKeywords": ["banarasi", "silk saree"],
        "variants": [{"color": "red"}, {"color": "maroon"}],
        "description": "Handwoven Banarasi saree with golden zari border.",
    }
    text = product_to_text(product)
    assert "Banarasi Silk Saree" in text
    assert "Women > Sarees" in text
    assert "wedding" in text
    assert "colors: red, maroon" in text
    assert "Handwoven Banarasi saree" in text


def test_product_to_text_handles_missing_fields():
    text = product_to_text({"name": "Plain Shirt"})
    assert text == "Plain Shirt"


def test_seller_to_text_includes_top_categories():
    seller = {
        "username": "mukura",
        "sellerProfile": {"shopName": "Mukura Silks", "bio": "Family-run saree store."},
        "categoryType": "Fashion",
    }
    text = seller_to_text(seller, top_categories=["Sarees", "Lehengas"])
    assert "mukura" in text
    assert "Mukura Silks" in text
    assert "sells: Sarees, Lehengas" in text
    assert "Family-run saree store." in text


def test_bit_to_text_includes_title_and_hashtags():
    bit = {
        "title": "Italian pants",
        "description": "Italian pants",
        "hashtags": ["#pants", "#italian"],
    }
    text = bit_to_text(bit)
    assert "Italian pants" in text
    assert "tags: pants italian" in text
    # description duplicate of title should not appear twice
    assert text.count("Italian pants") == 1


def test_bit_to_text_handles_missing_fields():
    text = bit_to_text({"title": "Quick look"})
    assert text == "Quick look"


def test_text_hash_is_stable_and_sensitive_to_change():
    h1 = text_hash("hello world")
    h2 = text_hash("hello world")
    h3 = text_hash("hello world!")
    assert h1 == h2
    assert h1 != h3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nAll tests passed.")
