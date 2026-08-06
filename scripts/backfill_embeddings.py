"""
backfill_embeddings.py — computes/refreshes the embedding vector for
every product, seller, and bit, so semantic search has vectors to search
over. Runs entirely locally (app/embeddings.py uses fastembed, not a
hosted API), so there's no rate limit or API key needed here.

ARCHITECTURE (Approach B): embeddings are NOT written onto the product/
seller/bit documents themselves. They live in separate companion
collections -- product_embeddings, seller_embeddings, bit_embeddings --
each document keyed by the SAME _id as the record it describes:

    product_embeddings: { _id: <products._id>, embedding: [...], embeddingHash: "...", updatedAt: ... }

This means the backfill only ever needs write access to these three
small, dedicated collections -- never to products/users/bits themselves.
That's a much narrower, easier-to-approve permission than write access
to live product/order data, and keeps the AI-search layer fully separable
from the core data model (removing the feature is just dropping 3
collections, no cleanup needed elsewhere). Search joins back to the
source collection at query time via $lookup (see app/vector_search.py).

Run manually for the first backfill:
    python scripts/backfill_embeddings.py

Then schedule it (see the `zatch-search-backfill` cron job in render.yaml)
so new/edited products/sellers/bits pick up vectors automatically. Docs
are skipped if their embeddingHash (in the companion collection) already
matches their current text, so re-runs are cheap.

Assumes `products.sellerId` references `users._id` (used to compute each
seller's top categories for their "sells: ..." embedding text). Adjust
`_top_categories()` below if your schema names this field differently.
"""
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import get_db
from app.embeddings import (
    embed_texts, product_to_text, seller_to_text, bit_to_text, text_hash,
    top_categories,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")

BATCH_SIZE = 100


def _existing_hashes(embeddings_coll) -> dict:
    """{_id: embeddingHash} for every doc already in a companion collection,
    used to decide which source records actually need re-embedding."""
    return {
        doc["_id"]: doc.get("embeddingHash")
        for doc in embeddings_coll.find({}, {"embeddingHash": 1})
    }


def _write_embeddings(embeddings_coll, docs, texts, vectors):
    now = datetime.now(timezone.utc)
    for doc, text, vector in zip(docs, texts, vectors):
        embeddings_coll.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "embedding": vector,
                "embeddingHash": text_hash(text),
                "updatedAt": now,
            }},
            upsert=True,
        )


def backfill_products(db):
    existing = _existing_hashes(db.product_embeddings)
    cursor = db.products.find({}, {
        "name": 1, "category": 1, "subCategory": 1, "tags": 1,
        "searchKeywords": 1, "variants": 1, "description": 1,
    })

    batch_docs, batch_texts = [], []
    updated = skipped = 0

    def flush():
        nonlocal updated
        if not batch_docs:
            return
        vectors = embed_texts(batch_texts, input_type="document")
        _write_embeddings(db.product_embeddings, batch_docs, batch_texts, vectors)
        updated += len(batch_docs)
        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        text = product_to_text(doc)
        if not text.strip():
            continue
        if existing.get(doc["_id"]) == text_hash(text):
            skipped += 1
            continue
        batch_docs.append(doc)
        batch_texts.append(text)
        if len(batch_docs) >= BATCH_SIZE:
            flush()
    flush()

    logger.info("Products: updated %d, skipped (unchanged) %d", updated, skipped)


def backfill_sellers(db):
    existing = _existing_hashes(db.seller_embeddings)
    # NOTE: filtering on sellerStatus == "active", not just field existence.
    # Every user document has a sellerStatus field by default ("buyer" for
    # regular customers, "rejected" for declined seller applications, "active"
    # for approved sellers) -- an existence check matches the entire user
    # base, not just real sellers. There is no downstream live-status filter
    # for sellers at query time (unlike products/bits), so this has to be
    # correct here.
    cursor = db.users.find(
        {"sellerStatus": "active"},
        {"username": 1, "sellerProfile": 1, "categoryType": 1},
    )

    batch_docs, batch_texts = [], []
    updated = skipped = 0

    def flush():
        nonlocal updated
        if not batch_docs:
            return
        vectors = embed_texts(batch_texts, input_type="document")
        _write_embeddings(db.seller_embeddings, batch_docs, batch_texts, vectors)
        updated += len(batch_docs)
        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        seller_categories = top_categories(db, doc["_id"])
        text = seller_to_text(doc, seller_categories)
        if not text.strip():
            continue
        if existing.get(doc["_id"]) == text_hash(text):
            skipped += 1
            continue
        batch_docs.append(doc)
        batch_texts.append(text)
        if len(batch_docs) >= BATCH_SIZE:
            flush()
    flush()

    logger.info("Sellers: updated %d, skipped (unchanged) %d", updated, skipped)


def backfill_bits(db):
    existing = _existing_hashes(db.bit_embeddings)
    cursor = db.bits.find({}, {"title": 1, "description": 1, "hashtags": 1})

    batch_docs, batch_texts = [], []
    updated = skipped = 0

    def flush():
        nonlocal updated
        if not batch_docs:
            return
        vectors = embed_texts(batch_texts, input_type="document")
        _write_embeddings(db.bit_embeddings, batch_docs, batch_texts, vectors)
        updated += len(batch_docs)
        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        text = bit_to_text(doc)
        if not text.strip():
            continue
        if existing.get(doc["_id"]) == text_hash(text):
            skipped += 1
            continue
        batch_docs.append(doc)
        batch_texts.append(text)
        if len(batch_docs) >= BATCH_SIZE:
            flush()
    flush()

    logger.info("Bits: updated %d, skipped (unchanged) %d", updated, skipped)


def main():
    db = get_db()
    logger.info("Starting embedding backfill (writing to companion collections only)...")
    backfill_products(db)
    backfill_sellers(db)
    backfill_bits(db)
    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
