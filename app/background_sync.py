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

from app.change_stream_listener import (
    _SYNC_HANDLERS, _watch_forever,
)
from app.db import get_db

logger = logging.getLogger("background_sync")

RECONCILE_INTERVAL_SECONDS = 3600  # safety-net pass once an hour while awake

# (source collection, embeddings collection, source filter). The filter
# mirrors the rule in the handlers: only active sellers get an embedding.
_RECONCILE_SPECS = [
    ("products", "product_embeddings", {}),
    ("users", "seller_embeddings", {"sellerStatus": "active"}),
    ("bits", "bit_embeddings", {}),
]


def reconcile(db) -> dict:
    """Embeds every source record that is missing/changed and deletes
    embeddings whose source is gone. Returns {source: orphans_removed}."""
    removed = {}
    for source, embeddings, source_filter in _RECONCILE_SPECS:
        handler = _SYNC_HANDLERS[source]
        ids = [d["_id"] for d in db[source].find(source_filter, {"_id": 1})]
        for doc_id in ids:
            try:
                handler(db, doc_id)  # no-ops when the text hash is unchanged
            except Exception:
                logger.exception("Reconcile: failed to sync %s _id=%s", source, doc_id)

        # Orphan cleanup. Skipped when the source came back empty, so a
        # wrong/empty read can never wipe a whole embeddings collection.
        orphans_removed = 0
        if ids:
            valid = set(ids)
            for d in db[embeddings].find({}, {"_id": 1}):
                if d["_id"] not in valid:
                    db[embeddings].delete_one({"_id": d["_id"]})
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
            removed = reconcile(db)
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
