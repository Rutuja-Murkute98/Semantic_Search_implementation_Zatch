"""
background_sync.py — keeps embeddings in sync from inside the web service.

Why this exists: Render's free tier has no background workers or cron
jobs, so the standalone listener (scripts/run_change_stream_listener.py)
and nightly backfill (scripts/backfill_embeddings.py) can't run there.
This module runs the same logic on daemon threads of the web process:

  1. Change-stream watchers (reused from app/change_stream_listener.py)
     embed new/edited products, sellers and bits within seconds.
  2. A reconcile pass runs at startup and then periodically. It embeds
     anything that is missing or whose text changed, and removes
     embeddings whose source record no longer exists. This is what closes
     any gap while the free instance was asleep or being redeployed.

Never raises into the web app: any failure is logged and the search API
keeps working (it just serves whatever embeddings already exist).

Set ENABLE_BACKGROUND_SYNC=0 to turn this off -- e.g. if you later run the
dedicated worker service, so the two don't do duplicate work.
"""
import logging
import os
import threading

from datetime import datetime, timezone

from app.change_stream_listener import (
    _SYNC_HANDLERS, _watch_forever,
)
from app.db import get_db
from app.embeddings import (
    embed_texts, product_to_text, seller_to_text, bit_to_text, text_hash,
    top_categories,
)

logger = logging.getLogger("background_sync")

RECONCILE_INTERVAL_SECONDS = 3600  # safety-net pass once an hour while awake

EMBED_BATCH_SIZE = 16

# (source collection, embeddings collection, source filter, text builder).
# The filter mirrors the rule in the change-stream handlers: only active
# sellers get an embedding.
def _product_text(db, doc):
    return product_to_text(doc)


def _seller_text(db, doc):
    return seller_to_text(doc, top_categories(db, doc["_id"]))


def _bit_text(db, doc):
    return bit_to_text(doc)


_RECONCILE_SPECS = [
    ("products", "product_embeddings", {}, _product_text),
    ("users", "seller_embeddings", {"sellerStatus": "active"}, _seller_text),
    ("bits", "bit_embeddings", {}, _bit_text),
]


def _priority(doc) -> int:
    """Live (active) products first -- those are the ones customers can
    actually see, so they should get embedded before drafts/inactive."""
    return 0 if doc.get("status") == "active" else 1


def reconcile(db, stop_event: threading.Event = None) -> dict:
    """Embeds every source record that is missing/changed (in batches) and
    deletes embeddings whose source is gone. Returns {source: orphans_removed}.

    Compares hashes in bulk first, so an up-to-date collection costs only a
    couple of reads instead of one round trip per document."""
    removed = {}
    for source, embeddings, source_filter, build_text in _RECONCILE_SPECS:
        docs = list(db[source].find(source_filter))
        docs.sort(key=_priority)
        existing = {d["_id"]: d.get("embeddingHash")
                    for d in db[embeddings].find({}, {"embeddingHash": 1})}

        todo = []
        for doc in docs:
            try:
                text = build_text(db, doc)
            except Exception:
                logger.exception("Reconcile: failed to build text for %s _id=%s", source, doc.get("_id"))
                continue
            if text.strip() and existing.get(doc["_id"]) != text_hash(text):
                todo.append((doc["_id"], text))

        for i in range(0, len(todo), EMBED_BATCH_SIZE):
            if stop_event is not None and stop_event.is_set():
                return removed
            batch = todo[i:i + EMBED_BATCH_SIZE]
            try:
                vectors = embed_texts([t for _, t in batch], input_type="document")
                now = datetime.now(timezone.utc)
                for (doc_id, text), vector in zip(batch, vectors):
                    db[embeddings].update_one(
                        {"_id": doc_id},
                        {"$set": {"embedding": vector, "embeddingHash": text_hash(text),
                                  "updatedAt": now}},
                        upsert=True,
                    )
            except Exception:
                logger.exception("Reconcile: failed to embed a batch for %s", source)
        if todo:
            logger.info("Reconcile: %s -- embedded %d of %d needing it.", source, len(todo), len(docs))

        # Orphan cleanup. Skipped when the source came back empty, so a
        # wrong/empty read can never wipe a whole embeddings collection.
        orphans_removed = 0
        if docs:
            valid = {d["_id"] for d in docs}
            for doc_id in existing:
                if doc_id not in valid:
                    db[embeddings].delete_one({"_id": doc_id})
                    orphans_removed += 1
        removed[source] = orphans_removed
    return removed


def run_forever(stop_event: threading.Event):
    db = get_db()

    # Start watching first, so nothing that changes during the reconcile
    # pass below can be missed.
    for name in _SYNC_HANDLERS:
        threading.Thread(
            target=_watch_forever, args=(db, name, stop_event),
            name=f"watch-{name}", daemon=True,
        ).start()

    while not stop_event.is_set():
        try:
            removed = reconcile(db, stop_event)
            logger.info("Embedding reconcile finished (orphans removed: %s).", removed)
        except Exception:
            logger.exception("Embedding reconcile failed; will retry next interval.")
        stop_event.wait(RECONCILE_INTERVAL_SECONDS)


def start():
    """Starts the sync threads (once). Returns the stop Event, or None if
    disabled via ENABLE_BACKGROUND_SYNC=0."""
    if os.getenv("ENABLE_BACKGROUND_SYNC", "1").strip().lower() in ("0", "false", "no"):
        logger.info("Background embedding sync disabled (ENABLE_BACKGROUND_SYNC).")
        return None
    stop_event = threading.Event()
    threading.Thread(
        target=run_forever, args=(stop_event,), name="embedding-sync", daemon=True,
    ).start()
    logger.info("Background embedding sync started.")
    return stop_event
