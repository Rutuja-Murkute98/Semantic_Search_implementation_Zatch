"""
vector_search.py — semantic ("meaning-based") retrieval leg.

Primary path: MongoDB Atlas $vectorSearch (plan §3, Phase 4).
Fallback (Plan B, plan §3): automatic in-process cosine similarity with
numpy, used the moment $vectorSearch raises (index missing, cluster isn't
on Atlas, etc). This means semantic search degrades gracefully instead of
breaking the API before the Atlas indexes even exist.
"""
import logging

from bson import ObjectId

logger = logging.getLogger("vector_search")

SIM_CUTOFF = 0.6              # drop matches below this similarity (avoid nonsense)
FALLBACK_SCAN_LIMIT = 20000    # cap for the Plan-B python-side scan

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


def _project_product(doc, score):
    price = doc.get("discountedPrice") or doc.get("price") or 0
    return {
        "type": "product",
        "id": oid_str(doc.get("_id")),
        "name": doc.get("name", ""),
        "price": price,
        "vectorScore": round(float(score), 4),
    }


def _project_seller(doc, score):
    sp = doc.get("sellerProfile") or {}
    pp = doc.get("profilePic") or {}
    pic = (pp.get("url", "") or pp.get("uri", "")) if isinstance(pp, dict) else ""
    return {
        "type": "seller",
        "id": oid_str(doc.get("_id")),
        "username": doc.get("username", ""),
        "shopName": sp.get("shopName", "") or sp.get("businessName", ""),
        "profilePic": pic,
        "followers": doc.get("followerCount", 0),
        "sellerStatus": doc.get("sellerStatus", ""),
        "vectorScore": round(float(score), 4),
    }


def semantic_products(db, query_vec, price_limit=None, limit=30) -> list:
    if not query_vec:
        return []

    vs_filter = {"status": "active"}
    if price_limit is not None:
        vs_filter["discountedPrice"] = {"$lte": price_limit}

    if _atlas_available(db.products, "products", "product_vector_index"):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "product_vector_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": vs_filter,
                }
            },
            {"$addFields": {"vectorSearchScore": {"$meta": "vectorSearchScore"}}},
        ]
        try:
            docs = list(db.products.aggregate(pipeline))
            docs = [d for d in docs if _is_live_product(d)]
            results = [
                _project_product(d, d.get("vectorSearchScore", 0))
                for d in docs
                if d.get("vectorSearchScore", 0) >= SIM_CUTOFF
            ]
            # Safety net for docs missing discountedPrice entirely.
            if price_limit is not None:
                results = [r for r in results if (r["price"] or 0) <= price_limit]
            return results
        except Exception as e:
            logger.warning(
                "Atlas $vectorSearch unavailable for products (falling back "
                "to in-process cosine similarity until the index exists): %s", e,
            )
            _atlas_unavailable["products"] = True

    mongo_filter = {"embedding": {"$exists": True}, "status": "active"}
    if price_limit is not None:
        mongo_filter["discountedPrice"] = {"$lte": price_limit}
    docs = list(db.products.find(mongo_filter).limit(FALLBACK_SCAN_LIMIT))
    docs = [d for d in docs if _is_live_product(d)]
    scored = _cosine_topk(query_vec, docs, limit=limit)
    return [_project_product(d, s) for s, d in scored]


def semantic_sellers(db, query_vec, limit=10) -> list:
    if not query_vec:
        return []

    if _atlas_available(db.users, "users", "seller_vector_index"):
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
        ]
        try:
            docs = list(db.users.aggregate(pipeline))
            return [
                _project_seller(d, d.get("vectorSearchScore", 0))
                for d in docs
                if d.get("vectorSearchScore", 0) >= SIM_CUTOFF
            ]
        except Exception as e:
            logger.warning(
                "Atlas $vectorSearch unavailable for sellers (falling back "
                "to in-process cosine similarity until the index exists): %s", e,
            )
            _atlas_unavailable["users"] = True

    docs = list(db.users.find({"embedding": {"$exists": True}}).limit(FALLBACK_SCAN_LIMIT))
    scored = _cosine_topk(query_vec, docs, limit=limit)
    return [_project_seller(d, s) for s, d in scored]


def _project_bit(doc, score):
    video = doc.get("video") or {}
    thumb = doc.get("thumbnail") or {}
    return {
        "type": "bit",
        "id": oid_str(doc.get("_id")),
        "title": doc.get("title", ""),
        "videoUrl": video.get("url", "") if isinstance(video, dict) else "",
        "thumbnailUrl": thumb.get("url", "") if isinstance(thumb, dict) else "",
        "likeCount": doc.get("likeCount", 0),
        "viewCount": doc.get("viewCount", 0),
        "vectorScore": round(float(score), 4),
    }


def semantic_bits(db, query_vec, limit=15) -> list:
    if not query_vec:
        return []

    if _atlas_available(db.bits, "bits", "bit_vector_index"):
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "bit_vector_index",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": limit * 10,
                    "limit": limit,
                }
            },
            {"$addFields": {"vectorSearchScore": {"$meta": "vectorSearchScore"}}},
        ]
        try:
            docs = list(db.bits.aggregate(pipeline))
            docs = [d for d in docs if _is_live_bit(d)]
            return [
                _project_bit(d, d.get("vectorSearchScore", 0))
                for d in docs
                if d.get("vectorSearchScore", 0) >= SIM_CUTOFF
            ]
        except Exception as e:
            logger.warning(
                "Atlas $vectorSearch unavailable for bits (falling back "
                "to in-process cosine similarity until the index exists): %s", e,
            )
            _atlas_unavailable["bits"] = True

    docs = list(db.bits.find({"embedding": {"$exists": True}}).limit(FALLBACK_SCAN_LIMIT))
    docs = [d for d in docs if _is_live_bit(d)]
    scored = _cosine_topk(query_vec, docs, limit=limit)
    return [_project_bit(d, s) for s, d in scored]
