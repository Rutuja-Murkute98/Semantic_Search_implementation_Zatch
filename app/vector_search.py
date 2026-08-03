"""
vector_search.py — semantic ("meaning-based") retrieval leg.

ARCHITECTURE (Approach B): embeddings live in dedicated companion
collections (product_embeddings, seller_embeddings, bit_embeddings),
never as fields on the actual products/users/bits documents. Each
companion doc shares its _id with the record it describes, so joining
back to the real product/seller/bit at query time is a simple $lookup on
_id. This means the search service never needs write access to the core
data collections at all -- only to these three small, dedicated ones.
See scripts/backfill_embeddings.py for the write side of this.

Primary path: MongoDB Atlas $vectorSearch on the companion collection,
then $lookup to fetch the real product/seller/bit fields needed for
filtering (status, price, live-ness) and display (name, price, etc).
Fallback (Plan B): automatic in-process cosine similarity with numpy,
used the moment $vectorSearch is unavailable (index missing, cluster
isn't on Atlas, etc). This means semantic search degrades gracefully
instead of breaking the API before the Atlas indexes even exist.
"""
import logging
import time

from bson import ObjectId

logger = logging.getLogger("vector_search")

SIM_CUTOFF = 0.6              # drop matches below this similarity (avoid nonsense)
FALLBACK_SCAN_LIMIT = 20000    # cap for the Plan-B python-side scan
OVERFETCH_MULTIPLIER = 3       # candidates requested from $vectorSearch, before
                                # the post-$lookup status/price $match can shrink
                                # the set below the caller's desired `limit`

# Cache for the Plan-B fallback fetch (all embedded docs, joined to their
# source record, in a collection). Fetching every seller/bit document
# over the network on every single search was the dominant cost when
# there's no Atlas vector index for that collection -- profiled at 2-7s
# for ~280 small documents on an Atlas M0 cluster, almost entirely
# network round-trip time, not the cosine-similarity computation itself.
# Embeddings only change when the backfill cron runs (every 6h), so a
# short TTL cache removes that cost from the request path between
# refreshes.
FALLBACK_CACHE_TTL_SECONDS = 300
_fallback_cache = {}   # {cache_key: (fetched_at, docs)}


def _cached_fallback_docs(embeddings_coll, source_coll, cache_key: str, match: dict = None) -> list:
    """Fetches every companion-collection doc, joined to its source record
    as `source`, via aggregation $lookup -- this is a standard MongoDB
    feature and works with or without Atlas Search/Vector Search enabled."""
    now = time.time()
    cached = _fallback_cache.get(cache_key)
    if cached and (now - cached[0]) < FALLBACK_CACHE_TTL_SECONDS:
        return cached[1]

    pipeline = [
        {"$lookup": {
            "from": source_coll.name,
            "localField": "_id",
            "foreignField": "_id",
            "as": "source",
        }},
        {"$unwind": "$source"},
    ]
    if match:
        pipeline.append({"$match": match})
    pipeline.append({"$limit": FALLBACK_SCAN_LIMIT})

    docs = list(embeddings_coll.aggregate(pipeline))
    _fallback_cache[cache_key] = (now, docs)
    return docs


# Per-process cache: once $vectorSearch is known unusable for a collection
# (index missing, not READY, or the query itself failed), stop retrying it
# on every request and go straight to the fallback. Cleared on process
# restart (e.g. after creating the index).
#
# IMPORTANT: Atlas silently returns an EMPTY result set (no exception) when
# $vectorSearch references an index name that doesn't exist -- it does not
# raise. Relying on try/except alone to detect "index missing" therefore
# doesn't work: the query "succeeds" with zero results forever, and the
# fallback never engages. So availability is checked explicitly up front
# via list_search_indexes(), not inferred from whether the query raised.
_atlas_unavailable = {"products": None, "users": None, "bits": None}


def oid_str(val):
    return str(val) if isinstance(val, ObjectId) else (val or "")


def _index_ready(coll, index_name: str) -> bool:
    try:
        indexes = list(coll.list_search_indexes(index_name))
    except Exception:
        return False
    return any(i.get("status") == "READY" for i in indexes)


def _atlas_available(coll, key: str, index_name: str) -> bool:
    """Resolves and caches (per process) whether an Atlas index is usable.
    Checked explicitly via list_search_indexes() rather than inferred from
    whether a query raised, since Atlas does not raise for a missing index
    -- see the _atlas_unavailable comment above."""
    if _atlas_unavailable[key] is None:
        _atlas_unavailable[key] = not _index_ready(coll, index_name)
    return not _atlas_unavailable[key]


def _is_live_product(p: dict) -> bool:
    """A product is only 'live' if active, in stock, and not already sold.
    Applied as a Python post-filter so it behaves identically whether
    results came from Atlas $vectorSearch or the cosine-similarity
    fallback, without needing totalStock/isSold declared as index filter
    fields."""
    return (
        p.get("status") == "active"
        and (p.get("totalStock") or 0) > 0
        and not p.get("isSold")
    )


def _is_live_bit(b: dict) -> bool:
    return bool(b.get("isActive"))


def _cosine_topk(query_vec, docs, vector_field="embedding", limit=30):
    import numpy as np

    q = np.array(query_vec, dtype="float32")
    q_norm = float(np.linalg.norm(q)) or 1.0

    scored = []
    for d in docs:
        vec = d.get(vector_field)
        if not vec:
            continue
        v = np.array(vec, dtype="float32")
        v_norm = float(np.linalg.norm(v)) or 1.0
        sim = float(np.dot(q, v) / (q_norm * v_norm))
        if sim >= SIM_CUTOFF:
            scored.append((sim, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:limit]


def _project_product(source, score):
    price = source.get("discountedPrice") or source.get("price") or 0
    return {
        "type": "product",
        "id": oid_str(source.get("_id")),
        "name": source.get("name", ""),
        "price": price,
        "vectorScore": round(float(score), 4),
    }


def _project_seller(source, score):
    sp = source.get("sellerProfile") or {}
    pp = source.get("profilePic") or {}
    pic = (pp.get("url", "") or pp.get("uri", "")) if isinstance(pp, dict) else ""
    return {
        "type": "seller",
        "id": oid_str(source.get("_id")),
        "username": source.get("username", ""),
        "shopName": sp.get("shopName", "") or sp.get("businessName", ""),
        "profilePic": pic,
        "followers": source.get("followerCount", 0),
        "sellerStatus": source.get("sellerStatus", ""),
        "vectorScore": round(float(score), 4),
    }


def _project_bit(source, score):
    video = source.get("video") or {}
    thumb = source.get("thumbnail") or {}
    return {
        "type": "bit",
        "id": oid_str(source.get("_id")),
        "title": source.get("title", ""),
        "videoUrl": video.get("url", "") if isinstance(video, dict) else "",
        "thumbnailUrl": thumb.get("url", "") if isinstance(thumb, dict) else "",
        "likeCount": source.get("likeCount", 0),
        "viewCount": source.get("viewCount", 0),
        "vectorScore": round(float(score), 4),
    }


def semantic_products(db, query_vec, price_limit=None, limit=30) -> list:
    if not query_vec:
        return []

    if _atlas_available(db.product_embeddings, "products", "product_vector_index"):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "product_vector_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": limit * 10,
                    "limit": limit * OVERFETCH_MULTIPLIER,
                }
            },
            {"$addFields": {"vectorSearchScore": {"$meta": "vectorSearchScore"}}},
            {"$lookup": {
                "from": "products", "localField": "_id", "foreignField": "_id", "as": "source",
            }},
            {"$unwind": "$source"},
            {"$match": {"source.status": "active"}},
        ]
        try:
            docs = list(db.product_embeddings.aggregate(pipeline))
            docs = [d for d in docs if _is_live_product(d["source"])]
            results = [
                _project_product(d["source"], d.get("vectorSearchScore", 0))
                for d in docs
                if d.get("vectorSearchScore", 0) >= SIM_CUTOFF
            ]
            if price_limit is not None:
                results = [r for r in results if (r["price"] or 0) <= price_limit]
            return results[:limit]
        except Exception as e:
            logger.warning(
                "Atlas $vectorSearch unavailable for products (falling back "
                "to in-process cosine similarity until the index exists): %s", e,
            )
            _atlas_unavailable["products"] = True

    docs = _cached_fallback_docs(
        db.product_embeddings, db.products, "products", {"source.status": "active"},
    )
    docs = [d for d in docs if _is_live_product(d["source"])]
    if price_limit is not None:
        docs = [d for d in docs
                if (d["source"].get("discountedPrice") or d["source"].get("price") or 0) <= price_limit]
    scored = _cosine_topk(query_vec, docs, limit=limit)
    return [_project_product(d["source"], s) for s, d in scored]


def semantic_sellers(db, query_vec, limit=10) -> list:
    if not query_vec:
        return []

    if _atlas_available(db.seller_embeddings, "users", "seller_vector_index"):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "seller_vector_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": limit * 10,
                    "limit": limit,
                }
            },
            {"$addFields": {"vectorSearchScore": {"$meta": "vectorSearchScore"}}},
            {"$lookup": {
                "from": "users", "localField": "_id", "foreignField": "_id", "as": "source",
            }},
            {"$unwind": "$source"},
        ]
        try:
            docs = list(db.seller_embeddings.aggregate(pipeline))
            return [
                _project_seller(d["source"], d.get("vectorSearchScore", 0))
                for d in docs
                if d.get("vectorSearchScore", 0) >= SIM_CUTOFF
            ]
        except Exception as e:
            logger.warning(
                "Atlas $vectorSearch unavailable for sellers (falling back "
                "to in-process cosine similarity until the index exists): %s", e,
            )
            _atlas_unavailable["users"] = True

    docs = _cached_fallback_docs(db.seller_embeddings, db.users, "users")
    scored = _cosine_topk(query_vec, docs, limit=limit)
    return [_project_seller(d["source"], s) for s, d in scored]


def semantic_bits(db, query_vec, limit=15) -> list:
    if not query_vec:
        return []

    if _atlas_available(db.bit_embeddings, "bits", "bit_vector_index"):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "bit_vector_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": limit * 10,
                    "limit": limit * OVERFETCH_MULTIPLIER,
                }
            },
            {"$addFields": {"vectorSearchScore": {"$meta": "vectorSearchScore"}}},
            {"$lookup": {
                "from": "bits", "localField": "_id", "foreignField": "_id", "as": "source",
            }},
            {"$unwind": "$source"},
        ]
        try:
            docs = list(db.bit_embeddings.aggregate(pipeline))
            docs = [d for d in docs if _is_live_bit(d["source"])]
            results = [
                _project_bit(d["source"], d.get("vectorSearchScore", 0))
                for d in docs
                if d.get("vectorSearchScore", 0) >= SIM_CUTOFF
            ]
            return results[:limit]
        except Exception as e:
            logger.warning(
                "Atlas $vectorSearch unavailable for bits (falling back "
                "to in-process cosine similarity until the index exists): %s", e,
            )
            _atlas_unavailable["bits"] = True

    docs = _cached_fallback_docs(db.bit_embeddings, db.bits, "bits")
    docs = [d for d in docs if _is_live_bit(d["source"])]
    scored = _cosine_topk(query_vec, docs, limit=limit)
    return [_project_bit(d["source"], s) for s, d in scored]
