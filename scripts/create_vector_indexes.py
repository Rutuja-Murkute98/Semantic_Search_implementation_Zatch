"""
create_vector_indexes.py — one-time setup: creates the two Atlas Vector
Search indexes semantic search depends on (plan §6, Phase 3).

Run once, after the backfill has produced `embedding` fields:
    python scripts/create_vector_indexes.py

Requires MongoDB Atlas (the free M0 tier supports Vector Search). Indexes
take about a minute to finish building; until they're queryable (or if
you're not on Atlas at all), app/vector_search.py automatically falls
back to in-process cosine similarity — so search keeps working either way.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo.operations import SearchIndexModel

from app.db import get_db

PRODUCT_INDEX = SearchIndexModel(
    definition={
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
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
            {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
        ]
    },
    name="seller_vector_index",
    type="vectorSearch",
)


def main():
    db = get_db()

    for name, coll, model in (
        ("product_vector_index", db.products, PRODUCT_INDEX),
        ("seller_vector_index", db.users, SELLER_INDEX),
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
