"""
create_text_indexes.py — one-time setup: creates the Atlas Search (fuzzy
text) index that app/text_search.py uses for typo-tolerant keyword search,
replacing the old hardcoded typo-correction dictionary.

Run once:
    python scripts/create_text_indexes.py

IMPORTANT — free tier (M0) index limit:
In practice this cluster supports only 2 Search/Vector Search indexes
total (observed empirically -- a 3rd index attempt is rejected regardless
of type, below the 3 the Atlas UI advertises for M0 in general). This
script creates one of those 2 slots for **products**, since that's the
specific capability requested (fuzzy matching for product search, e.g.
correcting "sari" -> "saree" without a dictionary) -- see
create_vector_indexes.py for how the other slot is allocated.

Sellers and bits still work correctly without a dedicated text index --
app/text_search.py automatically falls back to plain regex matching for
them. If you upgrade past M0, or free up a slot, define SELLER_TEXT_INDEX
/ BIT_TEXT_INDEX below the same way (already defined) and create them too
for full fuzzy coverage.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo.operations import SearchIndexModel

from app.db import get_db

PRODUCT_TEXT_INDEX = SearchIndexModel(
    definition={
        "mappings": {
            "dynamic": False,
            "fields": {
                "name": {"type": "string"},
                "category": {"type": "string"},
                "subCategory": {"type": "string"},
                "tags": {"type": "string"},
                "searchKeywords": {"type": "string"},
                "variants": {"type": "document", "fields": {"color": {"type": "string"}}},
            },
        }
    },
    name="product_text_index",
    type="search",
)

# Defined for completeness -- not created by default due to the M0 3-index
# cap. Create manually (coll.create_search_index(model=...)) if a slot
# frees up or the cluster is upgraded.
SELLER_TEXT_INDEX = SearchIndexModel(
    definition={
        "mappings": {
            "dynamic": False,
            "fields": {
                "username": {"type": "string"},
                "sellerProfile": {"type": "document", "fields": {
                    "shopName": {"type": "string"},
                    "businessName": {"type": "string"},
                }},
            },
        }
    },
    name="seller_text_index",
    type="search",
)

BIT_TEXT_INDEX = SearchIndexModel(
    definition={
        "mappings": {
            "dynamic": False,
            "fields": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "hashtags": {"type": "string"},
            },
        }
    },
    name="bit_text_index",
    type="search",
)


def main():
    db = get_db()

    try:
        existing = list(db.products.list_search_indexes("product_text_index"))
    except Exception as e:
        print(f"Could not list search indexes on products — is this cluster on Atlas? ({e})")
        return

    if existing:
        print("product_text_index already exists on products, skipping.")
    else:
        db.products.create_search_index(model=PRODUCT_TEXT_INDEX)
        print("Requested creation of product_text_index on products.")

    print(
        "\nNote: seller_text_index and bit_text_index were NOT created "
        "(this cluster's practical limit is 2 Search/Vector Search indexes "
        "total, both now used: product_vector_index + product_text_index). "
        "Seller and bit keyword search still work correctly via the "
        "automatic regex fallback in app/text_search.py -- just without "
        "Atlas's fuzzy typo tolerance until a slot is freed up or the "
        "cluster is upgraded."
    )
    print("Done. The index may take ~1 minute to finish building in Atlas.")


if __name__ == "__main__":
    main()
