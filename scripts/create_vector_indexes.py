"""
create_vector_indexes.py — one-time setup: creates the Atlas Vector Search
indexes semantic search depends on (plan §6, Phase 3).

Run once, after the backfill has produced `embedding` fields:
    python scripts/create_vector_indexes.py

Requires MongoDB Atlas (the free M0 tier supports Vector Search). Indexes
take about a minute to finish building; until they're queryable (or if
you're not on Atlas at all), app/vector_search.py automatically falls
back to in-process cosine similarity — so search keeps working either way.

INDEX BUDGET ON M0 (free tier): in practice this cluster only supports 2
Search/Vector Search indexes total, not the 3 the Atlas UI advertises for
M0 in general (observed empirically -- a 3rd index creation attempt was
rejected with "maximum number of FTS indexes has been reached" both times
it was tried, regardless of whether the 3rd was another vector index or
the text index from create_text_indexes.py). With product_text_index
already using one slot for fuzzy product search, only ONE of
PRODUCT_INDEX / SELLER_INDEX / BIT_INDEX below can also be created.
product_vector_index is enabled by default since product search is the
primary use case; seller and bit semantic search still work correctly via
the automatic cosine-similarity fallback, just not Atlas-accelerated.
Swap which index is enabled below, or upgrade past M0, to change this.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo.operations import SearchIndexModel

from app.db import get_db

PRODUCT_INDEX = SearchIndexModel(
    definition={
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
            {"type": "filter", "path": "status"},
            {"type": "filter", "path": "discountedPrice"},
        ]
    },
    name="product_vector_index",
    type="vectorSearch",
)

SELLER_INDEX = SearchIndexModel(
    definition={
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
        ]
    },
    name="seller_vector_index",
    type="vectorSearch",
)

BIT_INDEX = SearchIndexModel(
    definition={
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine"},
        ]
    },
    name="bit_vector_index",
    type="vectorSearch",
)


def main():
    db = get_db()

    # See INDEX BUDGET note above -- only one of these three should be
    # enabled at a time alongside product_text_index on M0.
    for name, coll, model in (
        ("product_vector_index", db.products, PRODUCT_INDEX),
        # ("bit_vector_index", db.bits, BIT_INDEX),
        # ("seller_vector_index", db.users, SELLER_INDEX),
    ):
        try:
            existing = list(coll.list_search_indexes(name))
        except Exception as e:
            print(f"Could not list search indexes on {coll.name} — "
                  f"is this cluster on Atlas? ({e})")
            continue

        if existing:
            print(f"{name} already exists on {coll.name}, skipping.")
            continue

        coll.create_search_index(model=model)
        print(f"Requested creation of {name} on {coll.name}.")

    print("Done. Indexes may take ~1 minute to finish building in Atlas.")


if __name__ == "__main__":
    main()
