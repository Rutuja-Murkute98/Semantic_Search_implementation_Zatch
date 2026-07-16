"""
backfill_embeddings.py — computes/refreshes the `embedding` vector for every
product and seller, so semantic search has vectors to search over.

Run manually for the first backfill:
    python scripts/backfill_embeddings.py

Then schedule it (see the `zatch-search-backfill` cron job in render.yaml)
so new/edited products and sellers pick up vectors automatically. Docs are
skipped if their `embeddingHash` already matches their current text, so
re-runs are cheap.

Assumes `products.sellerId` references `users._id` (used to compute each
seller's top categories for their "sells: ..." embedding text). Adjust
`_top_categories()` below if your schema names this field differently.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_db
from app.embeddings import embed_texts, product_to_text, seller_to_text, text_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")

BATCH_SIZE = 100


def _needs_embedding(doc: dict, text: str) -> bool:
    return doc.get("embeddingHash") != text_hash(text)


def backfill_products(db):
    cursor = db.products.find({}, {
        "name": 1, "category": 1, "subCategory": 1, "tags": 1,
        "searchKeywords": 1, "variants": 1, "description": 1,
        "embeddingHash": 1,
    })

    batch_docs, batch_texts = [], []
    updated = skipped = 0

    def flush():
        nonlocal updated
        if not batch_docs:
            return
        vectors = embed_texts(batch_texts, input_type="document")
        for doc, text, vector in zip(batch_docs, batch_texts, vectors):
            db.products.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": vector, "embeddingHash": text_hash(text)}},
            )
        updated += len(batch_docs)
        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        text = product_to_text(doc)
        if not text.strip():
            continue
        if not _needs_embedding(doc, text):
            skipped += 1
            continue
        batch_docs.append(doc)
        batch_texts.append(text)
        if len(batch_docs) >= BATCH_SIZE:
            flush()
    flush()

    logger.info("Products: updated %d, skipped (unchanged) %d", updated, skipped)


def _top_categories(db, seller_id, limit=5) -> list:
    pipeline = [
        {"$match": {"sellerId": seller_id, "status": "active"}},
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
    ]
    return [d["_id"] for d in db.products.aggregate(pipeline) if d.get("_id")]


def backfill_sellers(db):
    cursor = db.users.find(
        {"sellerStatus": {"$exists": True}},
        {"username": 1, "sellerProfile": 1, "categoryType": 1, "embeddingHash": 1},
    )

    batch_docs, batch_texts = [], []
    updated = skipped = 0

    def flush():
        nonlocal updated
        if not batch_docs:
            return
        vectors = embed_texts(batch_texts, input_type="document")
        for doc, text, vector in zip(batch_docs, batch_texts, vectors):
            db.users.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": vector, "embeddingHash": text_hash(text)}},
            )
        updated += len(batch_docs)
        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        top_categories = _top_categories(db, doc["_id"])
        text = seller_to_text(doc, top_categories)
        if not text.strip():
            continue
        if not _needs_embedding(doc, text):
            skipped += 1
            continue
        batch_docs.append(doc)
        batch_texts.append(text)
        if len(batch_docs) >= BATCH_SIZE:
            flush()
    flush()

    logger.info("Sellers: updated %d, skipped (unchanged) %d", updated, skipped)


def main():
    if not os.getenv("VOYAGE_API_KEY"):
        logger.error("VOYAGE_API_KEY not set — aborting.")
        sys.exit(1)

    db = get_db()
    logger.info("Starting embedding backfill...")
    backfill_products(db)
    backfill_sellers(db)
    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
