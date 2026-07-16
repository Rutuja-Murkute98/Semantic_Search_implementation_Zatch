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

# Per-process cache: once $vectorSearch fails for a collection (e.g. index
# doesn't exist yet), stop retrying it on every request and go straight to
# the fallback. Cleared on process restart (e.g. after creating the index).
_atlas_unavailable = {"products": False, "users": False}


def oid_str(val):
    return str(val) if isinstance(val, ObjectId) else (val or "")


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

    if not _atlas_unavailable["products"]:
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
    scored = _cosine_topk(query_vec, docs, limit=limit)
    return [_project_product(d, s) for s, d in scored]


def semantic_sellers(db, query_vec, limit=10) -> list:
    if not query_vec:
        return []

    if not _atlas_unavailable["users"]:
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
