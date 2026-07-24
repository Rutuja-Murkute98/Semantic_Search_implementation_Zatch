"""
text_search.py — keyword retrieval via MongoDB Atlas Search fuzzy text
matching, replacing the old hardcoded typo-correction dictionary.

Why: a maintained dictionary of misspellings (`sareee` -> `saree`, ...)
can never cover every typo, and doesn't help at all with words nobody
thought to add. Atlas Search's `fuzzy` option on the `text` operator
tolerates typos natively (Levenshtein-distance based), against the real
catalog text, with no dictionary to maintain.

Falls back automatically to plain (non-fuzzy) regex matching if the Atlas
Search index isn't available yet — same fallback philosophy as
vector_search.py. Live-status filtering (§ live products / active bits)
is applied as a Python-side post-filter in both paths, so it behaves
identically regardless of which retrieval path served the candidates.
"""
import logging
import re

from rapidfuzz import fuzz

logger = logging.getLogger("text_search")

# Per-process cache, same pattern as vector_search.py's _atlas_unavailable.
# None = not yet resolved. IMPORTANT: Atlas silently returns an EMPTY
# result set (no exception) when $search references an index name that
# doesn't exist -- it does not raise. So availability is checked
# explicitly via list_search_indexes() the first time each collection is
# used, not inferred from whether the query raised (see _atlas_available
# in vector_search.py for the same fix, applied there for the same reason).
_atlas_text_unavailable = {"products": None, "users": None, "bits": None}

def fuzzy_options(word: str) -> dict:
    """Atlas Search only allows maxEdits of 1 or 2. 2 edits on a short
    word (<=4 chars, e.g. "sari") means up to half the characters can
    differ, matching unrelated words -- so short words get the tighter
    1-edit tolerance.

    For longer words, edit distance alone isn't enough: "pants" fuzzy-
    matches both "pattu" (edit distance 2) and "plants" (edit distance
    1) with maxEdits=2, pulling completely unrelated products into
    results. But tightening maxEdits down to 1 for these words is too
    aggressive -- it breaks legitimate 2-edit spelling variants that
    matter a lot in this catalog, like "kurta" -> "kurtis".

    prefixLength -- requiring that many leading characters to match
    exactly before fuzziness applies to the rest -- resolves this
    without that tradeoff: "pants" vs "pattu" and "pants" vs "plants"
    both diverge within the first 3 characters ("pan" vs "pat"/"pla"),
    so raising prefixLength to 3 for longer words rejects both, while
    "kurta" vs "kurtis" still share "kur" and remain a valid fuzzy
    match at maxEdits=2."""
    length = len(word)
    return {
        "maxEdits": 1 if length <= 4 else 2,
        "prefixLength": 3 if length > 4 else 1,
    }


def _index_ready(coll, index_name: str) -> bool:
    try:
        indexes = list(coll.list_search_indexes(index_name))
    except Exception:
        return False
    return any(i.get("status") == "READY" for i in indexes)


def _atlas_available(coll, key: str, index_name: str) -> bool:
    if _atlas_text_unavailable[key] is None:
        _atlas_text_unavailable[key] = not _index_ready(coll, index_name)
    return not _atlas_text_unavailable[key]


def is_live_product(p: dict) -> bool:
    """A product is only 'live' if active, in stock, and not already sold."""
    return (
        p.get("status") == "active"
        and (p.get("totalStock") or 0) > 0
        and not p.get("isSold")
    )


def is_live_bit(b: dict) -> bool:
    return bool(b.get("isActive"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _business_score(p: dict) -> float:
    score = 0.0
    score += min(p.get("likeCount", 0), 50) * 0.02
    score += min(p.get("saveCount", 0), 50) * 0.02
    score += min(p.get("viewCount", 0), 200) * 0.005
    return score


# ─────────────────────────────────────────────
# PRODUCTS
# ─────────────────────────────────────────────

def _product_search_pipeline(tokens, colors, main_term, limit):
    """
    main_term is required (compound.must): a document must actually match
    the product noun to qualify as a keyword result at all. Every other
    token and colour is compound.should -- an optional scoring boost
    among documents that already qualified, not an independent way in.

    Without this split, a modifier word alone could qualify a completely
    different product: searching "cotton pants" with all tokens as
    equally-weighted "should" clauses let any product merely containing
    "cotton" (a cotton kurti, a cotton saree) into the keyword results,
    which then out-ranked genuinely pants-related items that only
    matched via the semantic leg -- the customer's actual intent
    ("pants") was being treated as equally optional as the modifier.
    """
    must = []
    should = []
    if main_term:
        must.append({
            "text": {
                "query": main_term,
                "path": ["name", "category", "subCategory", "tags", "searchKeywords"],
                "fuzzy": fuzzy_options(main_term),
                "score": {"boost": {"value": 3}},
            }
        })
    for tok in tokens:
        if tok == main_term:
            continue
        should.append({
            "text": {"query": tok, "path": ["name", "tags"], "fuzzy": fuzzy_options(tok)}
        })
    for color in colors:
        should.append({
            "text": {"query": color, "path": ["name", "variants.color"], "fuzzy": fuzzy_options(color)}
        })

    compound = {}
    if must:
        compound["must"] = must
    if should:
        compound["should"] = should
        if not must:
            compound["minimumShouldMatch"] = 1

    return [
        {"$search": {"index": "product_text_index", "compound": compound}},
        {"$limit": limit},
        {"$addFields": {"textScore": {"$meta": "searchScore"}}},
    ]


def _product_regex_fallback(db, tokens, colors, main_term):
    """No fuzziness here -- this is the degraded path used only when Atlas
    Search itself is unavailable. The semantic leg covers meaning/typos
    the rest of the time.

    main_term must match -- same reasoning as _product_search_pipeline's
    compound.must: without it, a document matching only a secondary
    token/colour (e.g. "cotton" for a "cotton pants" search) would
    qualify as a keyword result on its own, even though it has nothing
    to do with what was actually being searched for."""
    if not main_term:
        return []
    pat = re.escape(main_term)
    query = {"$or": [
        {"name":           {"$regex": pat, "$options": "i"}},
        {"subCategory":    {"$regex": pat, "$options": "i"}},
        {"category":       {"$regex": pat, "$options": "i"}},
        {"tags":           {"$elemMatch": {"$regex": pat, "$options": "i"}}},
        {"searchKeywords": {"$elemMatch": {"$regex": pat, "$options": "i"}}},
    ]}
    return list(db.products.find(query))


def _fallback_match_score(p, tokens, colors, main_term) -> float:
    name = _normalize(p.get("name") or "")
    if main_term and main_term in name:
        base = 10.0 + (5.0 if name.startswith(main_term) else 0.0)
    elif main_term:
        base = fuzz.partial_ratio(main_term, name) / 100 * 4.0
    else:
        base = 0.0
    for tok in tokens:
        if tok in name:
            base += 2.0
    return base


def search_products(db, tokens: list, colors: set, main_term: str, price_limit, limit: int = 50) -> list:
    if not tokens:
        return []

    used_atlas = False
    if _atlas_available(db.products, "products", "product_text_index"):
        try:
            pipeline = _product_search_pipeline(tokens, colors, main_term, limit)
            raw = list(db.products.aggregate(pipeline))
            used_atlas = True
        except Exception as e:
            logger.warning(
                "Atlas $search unavailable for products (falling back to "
                "plain regex matching until the text index exists): %s", e,
            )
            _atlas_text_unavailable["products"] = True
            raw = _product_regex_fallback(db, tokens, colors, main_term)
    else:
        raw = _product_regex_fallback(db, tokens, colors, main_term)

    # Hard filters, applied identically regardless of retrieval path.
    raw = [p for p in raw if is_live_product(p)]
    if price_limit is not None:
        raw = [p for p in raw
               if (p.get("discountedPrice") or p.get("price") or 0) <= price_limit]

    if used_atlas:
        scored = sorted(
            ((p.get("textScore", 0) + _business_score(p), p) for p in raw),
            key=lambda x: x[0], reverse=True,
        )
    else:
        scored = sorted(
            ((_fallback_match_score(p, tokens, colors, main_term) + _business_score(p), p) for p in raw),
            key=lambda x: x[0], reverse=True,
        )

    results = []
    for _, p in scored:
        price = p.get("discountedPrice") or p.get("price") or 0
        results.append({
            "type": "product",
            "id": str(p.get("_id")),
            "name": p.get("name", ""),
            "price": price,
        })
    return results


# ─────────────────────────────────────────────
# SELLERS
# ─────────────────────────────────────────────

def _seller_search_pipeline(tokens, limit):
    should = [
        {"text": {
            "query": tok,
            "path": ["username", "sellerProfile.shopName", "sellerProfile.businessName"],
            "fuzzy": fuzzy_options(tok),
        }}
        for tok in tokens
    ]
    return [
        {"$search": {
            "index": "seller_text_index",
            "compound": {"should": should, "minimumShouldMatch": 1},
        }},
        {"$limit": limit},
        {"$addFields": {"textScore": {"$meta": "searchScore"}}},
    ]


def _seller_regex_fallback(db, tokens):
    clauses = []
    for tok in tokens:
        pat = re.escape(tok)
        clauses += [
            {"username":                   {"$regex": pat, "$options": "i"}},
            {"sellerProfile.shopName":     {"$regex": pat, "$options": "i"}},
            {"sellerProfile.businessName": {"$regex": pat, "$options": "i"}},
        ]
    if not clauses:
        return []
    return list(db.users.find(
        {"$or": clauses},
        {"_id": 1, "username": 1, "sellerProfile": 1, "sellerStatus": 1,
         "profilePic": 1, "followerCount": 1, "categoryType": 1},
    ))


def _project_seller(s: dict) -> dict:
    sp = s.get("sellerProfile") or {}
    pp = s.get("profilePic") or {}
    pic = (pp.get("url", "") or pp.get("uri", "")) if isinstance(pp, dict) else ""
    return {
        "type": "seller",
        "id": str(s["_id"]),
        "username": s.get("username", ""),
        "shopName": sp.get("shopName", "") or sp.get("businessName", ""),
        "profilePic": pic,
        "followers": s.get("followerCount", 0),
        "sellerStatus": s.get("sellerStatus", ""),
    }


def search_sellers(db, tokens: list, limit: int = 20) -> list:
    if not tokens:
        return []

    used_atlas = False
    if _atlas_available(db.users, "users", "seller_text_index"):
        try:
            raw = list(db.users.aggregate(_seller_search_pipeline(tokens, limit)))
            used_atlas = True
        except Exception as e:
            logger.warning(
                "Atlas $search unavailable for sellers (falling back to "
                "plain regex matching until the text index exists): %s", e,
            )
            _atlas_text_unavailable["users"] = True
            raw = _seller_regex_fallback(db, tokens)
    else:
        raw = _seller_regex_fallback(db, tokens)

    def _score(s):
        if used_atlas:
            return s.get("textScore", 0)
        uname = _normalize(s.get("username") or "")
        sp = s.get("sellerProfile") or {}
        shop = _normalize(sp.get("shopName", "") or sp.get("businessName", ""))
        total = 0.0
        for tok in tokens:
            if tok in uname or tok in shop:
                total += 10
            else:
                total += fuzz.partial_ratio(tok, uname + " " + shop) / 10
        return total

    scored = sorted(((_score(s), s) for s in raw), key=lambda x: x[0], reverse=True)
    return [_project_seller(s) for _, s in scored]


# ─────────────────────────────────────────────
# BITS
# ─────────────────────────────────────────────

def _bit_search_pipeline(tokens, limit):
    should = [
        {"text": {
            "query": tok,
            "path": ["title", "description", "hashtags"],
            "fuzzy": fuzzy_options(tok),
        }}
        for tok in tokens
    ]
    return [
        {"$search": {
            "index": "bit_text_index",
            "compound": {"should": should, "minimumShouldMatch": 1},
        }},
        {"$limit": limit},
        {"$addFields": {"textScore": {"$meta": "searchScore"}}},
    ]


def _bit_regex_fallback(db, tokens):
    clauses = []
    for tok in tokens:
        pat = re.escape(tok)
        clauses += [
            {"title":       {"$regex": pat, "$options": "i"}},
            {"description": {"$regex": pat, "$options": "i"}},
            {"hashtags":    {"$elemMatch": {"$regex": pat, "$options": "i"}}},
        ]
    if not clauses:
        return []
    return list(db.bits.find({"$or": clauses}))


def _project_bit(b: dict) -> dict:
    video = b.get("video") or {}
    thumb = b.get("thumbnail") or {}
    return {
        "type": "bit",
        "id": str(b["_id"]),
        "title": b.get("title", ""),
        "videoUrl": video.get("url", "") if isinstance(video, dict) else "",
        "thumbnailUrl": thumb.get("url", "") if isinstance(thumb, dict) else "",
        "likeCount": b.get("likeCount", 0),
        "viewCount": b.get("viewCount", 0),
    }


def search_bits(db, tokens: list, limit: int = 30) -> list:
    if not tokens:
        return []

    used_atlas = False
    if _atlas_available(db.bits, "bits", "bit_text_index"):
        try:
            raw = list(db.bits.aggregate(_bit_search_pipeline(tokens, limit)))
            used_atlas = True
        except Exception as e:
            logger.warning(
                "Atlas $search unavailable for bits (falling back to "
                "plain regex matching until the text index exists): %s", e,
            )
            _atlas_text_unavailable["bits"] = True
            raw = _bit_regex_fallback(db, tokens)
    else:
        raw = _bit_regex_fallback(db, tokens)

    raw = [b for b in raw if is_live_bit(b)]

    def _score(b):
        if used_atlas:
            return b.get("textScore", 0)
        title = _normalize(b.get("title") or "")
        total = 0.0
        for tok in tokens:
            if tok in title:
                total += 10
        return total + min(b.get("likeCount", 0), 50) * 0.02

    scored = sorted(((_score(b), b) for b in raw), key=lambda x: x[0], reverse=True)
    return [_project_bit(b) for _, b in scored]
