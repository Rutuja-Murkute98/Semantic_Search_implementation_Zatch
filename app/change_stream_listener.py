"""
change_stream_listener.py — real-time, event-driven embedding sync.

Watches products/users/bits for inserts/updates/deletes via MongoDB
Change Streams and keeps the companion embedding collections
(product_embeddings/seller_embeddings/bit_embeddings) in sync within
seconds of the change happening -- no manual re-run, no fixed polling
delay, no snapshot.

This is the automatic counterpart to scripts/backfill_embeddings.py,
which stays in place as a periodic safety-net reconciliation pass (see
the cron job in render.yaml) in case an event is ever missed -- e.g. a
brief gap between this listener stopping and its replacement starting
during a deploy.

Resume tokens are persisted in a small state collection
(`_embedding_sync_state`, one tiny document per watched collection) so a
restart resumes exactly where it left off instead of missing events or
reprocessing everything from scratch. This is the only collection this
module ever writes to besides the 3 embedding companions -- products,
users, and bits are only ever read here, never modified.

Run standalone:
    python scripts/run_change_stream_listener.py
"""
import logging
import threading
import time
from datetime import datetime, timezone

from pymongo.errors import PyMongoError

from app.db import get_db
from app.embeddings import (
    embed_texts, product_to_text, seller_to_text, bit_to_text, text_hash,
    top_categories,
)

logger = logging.getLogger("change_stream_listener")

STATE_COLLECTION = "_embedding_sync_state"
IDLE_POLL_SECONDS = 0.5      # brief pause when try_next() returns nothing
RECONNECT_DELAY_SECONDS = 5  # backoff before retrying a dropped stream


def _get_resume_token(db, watcher_name: str):
    doc = db[STATE_COLLECTION].find_one({"_id": watcher_name})
    return doc.get("resumeToken") if doc else None


def _save_resume_token(db, watcher_name: str, token):
    db[STATE_COLLECTION].update_one(
        {"_id": watcher_name},
        {"$set": {"resumeToken": token, "updatedAt": datetime.now(timezone.utc)}},
        upsert=True,
    )


def _upsert_if_changed(embeddings_coll, doc_id, text: str):
    """Shared write path for all 3 sync functions below: skip the (real,
    non-free) embedding computation if the text-relevant fields haven't
    actually changed since last time -- an update event can fire for
    fields we don't embed at all (viewCount, likeCount, etc.)."""
    if not text.strip():
        return
    h = text_hash(text)
    existing = embeddings_coll.find_one({"_id": doc_id}, {"embeddingHash": 1})
    if existing and existing.get("embeddingHash") == h:
        return
    vector = embed_texts([text], input_type="document")[0]
    embeddings_coll.update_one(
        {"_id": doc_id},
        {"$set": {
            "embedding": vector,
            "embeddingHash": h,
            "updatedAt": datetime.now(timezone.utc),
        }},
        upsert=True,
    )
    logger.info("Synced embedding for %s", doc_id)


def _sync_product(db, doc_id):
    doc = db.products.find_one({"_id": doc_id})
    if not doc:
        if db.product_embeddings.delete_one({"_id": doc_id}).deleted_count:
            logger.info("Product %s deleted -- removed its embedding.", doc_id)
        return
    _upsert_if_changed(db.product_embeddings, doc_id, product_to_text(doc))


def _sync_seller(db, doc_id):
    doc = db.users.find_one({"_id": doc_id})
    # Same rule as the backfill script: only users with sellerStatus ==
    # "active" are real, approved sellers. A user with no active seller
    # status (deleted, deactivated, still just "buyer", or "rejected")
    # should not have a seller embedding -- there is no downstream
    # live-status filter for sellers at query time, unlike products/bits.
    if not doc or doc.get("sellerStatus") != "active":
        if db.seller_embeddings.delete_one({"_id": doc_id}).deleted_count:
            logger.info("Seller %s no longer active -- removed its embedding.", doc_id)
        return
    seller_categories = top_categories(db, doc_id)
    _upsert_if_changed(db.seller_embeddings, doc_id, seller_to_text(doc, seller_categories))


def _sync_bit(db, doc_id):
    doc = db.bits.find_one({"_id": doc_id})
    if not doc:
        if db.bit_embeddings.delete_one({"_id": doc_id}).deleted_count:
            logger.info("Bit %s deleted -- removed its embedding.", doc_id)
        return
    _upsert_if_changed(db.bit_embeddings, doc_id, bit_to_text(doc))


_SYNC_HANDLERS = {
    "products": _sync_product,
    "users": _sync_seller,
    "bits": _sync_bit,
}


def _watch_forever(db, collection_name: str, stop_event: threading.Event):
    """Runs one collection's change-stream watch loop until stop_event is
    set, reconnecting automatically (from the last saved resume token) if
    the stream ever drops."""
    handler = _SYNC_HANDLERS[collection_name]

    while not stop_event.is_set():
        resume_token = _get_resume_token(db, collection_name)
        try:
            with db[collection_name].watch(
                resume_after=resume_token, full_document="updateLookup"
            ) as stream:
                logger.info(
                    "Watching '%s' for changes%s.",
                    collection_name,
                    " (resuming from saved position)" if resume_token else "",
                )
                while stream.alive and not stop_event.is_set():
                    change = stream.try_next()
                    if change is None:
                        time.sleep(IDLE_POLL_SECONDS)
                        continue

                    doc_id = change["documentKey"]["_id"]
                    try:
                        handler(db, doc_id)
                    except Exception:
                        logger.exception(
                            "Failed to sync %s _id=%s -- will retry on the "
                            "next scheduled backfill safety-net pass.",
                            collection_name, doc_id,
                        )
                    # Save progress after every event (processed or not) so
                    # a restart never reprocesses the same event twice.
                    _save_resume_token(db, collection_name, stream.resume_token)
        except PyMongoError:
            logger.exception(
                "Change stream on '%s' dropped -- reconnecting in %ss.",
                collection_name, RECONNECT_DELAY_SECONDS,
            )
            time.sleep(RECONNECT_DELAY_SECONDS)


def run(stop_event: threading.Event = None):
    """Starts one watcher thread per collection (products/users/bits) and
    blocks until stop_event is set (or forever, if none is given -- the
    normal case when run as a standalone process)."""
    db = get_db()
    stop_event = stop_event or threading.Event()

    threads = [
        threading.Thread(
            target=_watch_forever, args=(db, name, stop_event),
            name=f"watch-{name}", daemon=True,
        )
        for name in _SYNC_HANDLERS
    ]
    for t in threads:
        t.start()

    logger.info("Change stream listener running -- watching products, users, bits.")
    for t in threads:
        t.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
