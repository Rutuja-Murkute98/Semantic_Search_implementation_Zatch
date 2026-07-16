"""
search.py — Zatch Search v5 (hybrid semantic + keyword)
=========================================================
- Keyword leg: regex/fuzzy search over name/tags/category (unchanged v4 logic)
- Semantic leg: embeds the query (Voyage AI) and runs $vectorSearch on
  products + sellers, so meaning-based queries like "wedding outfit" work
  even when no product literally contains that word
- Both legs are merged with Reciprocal Rank Fusion (app/hybrid.py)
- Graceful degradation: if the embedding call fails or is slow, search
  silently falls back to keyword-only — it never 500s because of that
- Seller search → returns seller info only (not their products)
- Output simplified: id, name, price only for products
- Typo correction expanded
- Smart routing: if query matches a seller name → show seller
  if query matches product terms → show products
  if both match → show both sections separately
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from unidecode import unidecode
from rapidfuzz import fuzz, process
from fastapi import APIRouter
from app.db import get_db
from app.embeddings import embed_query
from app.vector_search import semantic_products, semantic_sellers
from app.hybrid import rrf_merge
from bson import ObjectId

router = APIRouter()
logger = logging.getLogger("search")

QUERY_EMBED_TIMEOUT_SECONDS = 2.0
_embed_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed")

# ─────────────────────────────────────────────
# TYPO CORRECTION TABLE
# ─────────────────────────────────────────────
CORRECTIONS = {
    # saree
    "sareee": "saree", "sarees": "saree", "sari": "saree", "saris": "saree",
    "saari": "saree", "sarree": "saree", "saarie": "saree",
    "saere": "saree", "srare": "saree",
    # dress
    "dres": "dress", "drses": "dress", "dresse": "dress", "drress": "dress",
    # shirt
    "shrit": "shirt", "shiirt": "shirt", "shir": "shirt",
    "tshirt": "t-shirt", "tshirts": "t-shirt", "tshit": "t-shirt",
    # jeans/pants
    "jeens": "jeans", "jean": "jeans", "jins": "jeans",
    "pant": "pants", "trouser": "trousers", "trousr": "trousers",
    # shoes/footwear
    "sheos": "shoes", "shoe": "shoes", "shose": "shoes", "shoees": "shoes",
    "sandal": "sandals", "sandle": "sandals", "sandels": "sandals",
    "slipper": "slippers", "slipers": "slippers",
    # watches
    "wach": "watch", "watche": "watch", "wtach": "watch",
    # Indian fashion
    "kurtis": "kurti", "kurtee": "kurti", "kurthi": "kurti",
    "dupata": "dupatta", "duaptta": "dupatta",
    "lehenga": "lehenga", "lehnga": "lehenga", "lahenga": "lehenga",
    "salwar": "salwar", "salvar": "salwar",
    "churidar": "churidar", "chudidar": "churidar",
    # jewellery
    "jewellry": "jewellery", "jewlery": "jewellery", "jwellery": "jewellery",
    "necklase": "necklace", "neckless": "necklace",
    "earing": "earring", "earings": "earrings", "earring": "earring",
    "bangel": "bangle", "bangels": "bangles",
    # bags
    "handbag": "handbag", "handabg": "handbag",
    "backpack": "backpack", "bakpack": "backpack",
    # electronics
    "mobil": "mobile", "moble": "mobile",
    "labtop": "laptop", "leptop": "laptop",
    # price words
    "belo": "below", "bellow": "below", "belw": "below",
    "undr": "under", "uder": "under",
}

COLORS = {
    "red", "blue", "green", "black", "white", "yellow", "pink", "purple",
    "orange", "grey", "gray", "brown", "beige", "maroon", "navy", "golden",
    "silver", "cream", "multicolor", "multi",
}

PRICE_STOP = {
    "under", "below", "less", "than", "upto", "up", "to", "within",
    "rs", "rupees", "only", "for", "a", "an", "the", "me", "show",
    "find", "get", "buy", "some",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unidecode(text or "").lower().strip())


def correct_word(word: str) -> str:
    """Correct a single word using lookup first, then fuzzy if needed."""
    if word in CORRECTIONS:
        return CORRECTIONS[word]
    # Only fuzzy-correct if word looks like a misspelling (not a name/brand)
    # Skip short words and words that look like names (uppercase start in original)
    if len(word) < 4:
        return word
    best = process.extractOne(word, list(CORRECTIONS.keys()), scorer=fuzz.ratio)
    if best and best[1] >= 80:
        return CORRECTIONS[best[0]]
    return word


def correct_query(query: str) -> str:
    words = normalize(query).split()
    return " ".join(correct_word(w) for w in words)


def extract_price_limit(query: str):
    patterns = [
        r"(?:under|below|less\s+than|upto|up\s+to|within)\s*₹?\s*(\d+)",
        r"₹?\s*(\d{3,5})\s*(?:only|rs|rupees|/-)?$",
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def extract_colors(query: str):
    return set(normalize(query).split()) & COLORS


def search_tokens(query: str) -> list:
    return [w for w in normalize(query).split()
            if w not in PRICE_STOP and not w.isdigit() and len(w) > 1]


def oid_str(val):
    return str(val) if isinstance(val, ObjectId) else (val or "")


# ─────────────────────────────────────────────
# SELLER SEARCH — returns seller profiles only
# ─────────────────────────────────────────────

def search_sellers(db, tokens: list) -> list:
    """
    Search users collection by username or shopName.
    Returns seller profile cards — NOT their products.
    """
    if not tokens:
        return []

    clauses = []
    for tok in tokens:
        pat = re.escape(tok)
        clauses += [
            {"username":                   {"$regex": pat, "$options": "i"}},
            {"sellerProfile.shopName":     {"$regex": pat, "$options": "i"}},
            {"sellerProfile.businessName": {"$regex": pat, "$options": "i"}},
        ]

    sellers = list(db.users.find(
        {"$or": clauses},
        {
            "_id": 1, "username": 1, "sellerProfile": 1,
            "sellerStatus": 1, "profilePic": 1,
            "followerCount": 1, "categoryType": 1,
        }
    ))

    results = []
    for s in sellers:
        sp  = s.get("sellerProfile") or {}
        pp  = s.get("profilePic") or {}
        pic = pp.get("url", "") or pp.get("uri", "") if isinstance(pp, dict) else ""

        # Score: exact username match ranks higher
        name_score = 0
        uname = normalize(s.get("username") or "")
        shop  = normalize(sp.get("shopName", "") or sp.get("businessName", ""))
        for tok in tokens:
            if tok in uname or tok in shop:
                name_score += 10
            else:
                name_score += fuzz.partial_ratio(tok, uname + " " + shop) / 10

        results.append({
            "_score":      name_score,
            "type":        "seller",
            "id":          oid_str(s["_id"]),
            "username":    s.get("username", ""),
            "shopName":    sp.get("shopName", "") or sp.get("businessName", ""),
            "profilePic":  pic,
            "followers":   s.get("followerCount", 0),
            "sellerStatus": s.get("sellerStatus", ""),
        })

    results.sort(key=lambda x: x["_score"], reverse=True)
    for r in results:
        del r["_score"]

    return results


# ─────────────────────────────────────────────
# PRODUCT SEARCH
# ─────────────────────────────────────────────

def build_product_query(tokens: list, colors: set, main_term: str) -> dict:
    clauses = []

    if main_term:
        pat = re.escape(main_term)
        # Deliberately excludes free-text `description` here: a stray word
        # anywhere in an unstructured description (e.g. "Running Colour" as
        # a saree fabric term) would otherwise qualify totally unrelated
        # products. Structured fields are a much more reliable qualifying
        # signal; meaning found only in free text is the semantic leg's job.
        clauses += [
            {"name":           {"$regex": pat, "$options": "i"}},
            {"subCategory":    {"$regex": pat, "$options": "i"}},
            {"category":       {"$regex": pat, "$options": "i"}},
            {"tags":           {"$elemMatch": {"$regex": pat, "$options": "i"}}},
            {"searchKeywords": {"$elemMatch": {"$regex": pat, "$options": "i"}}},
        ]

    for tok in tokens:
        if tok == main_term:
            continue
        pat = re.escape(tok)
        clauses += [
            {"name": {"$regex": pat, "$options": "i"}},
            {"tags": {"$elemMatch": {"$regex": pat, "$options": "i"}}},
        ]

    for color in colors:
        cpat = re.escape(color)
        clauses += [
            {"variants.color": {"$regex": cpat, "$options": "i"}},
            {"name":           {"$regex": cpat, "$options": "i"}},
        ]

    return {"$or": clauses} if clauses else {}


def score_product(p: dict, tokens: list, colors: set, main_term: str) -> float:
    name  = normalize(p.get("name") or "")
    sub   = normalize(p.get("subCategory") or "")
    cat   = normalize(p.get("category") or "")
    desc  = normalize(p.get("description") or "")[:300]
    tags  = normalize(" ".join(p.get("tags") or []))
    kws   = normalize(" ".join(p.get("searchKeywords") or []))
    vcols = " ".join(normalize(v.get("color","")) for v in (p.get("variants") or []))
    full  = f"{name} {sub} {cat} {tags} {kws} {desc} {vcols}"

    score = 0.0

    if main_term:
        if main_term in name:
            score += 10.0
            if name.startswith(main_term):
                score += 5.0
        score += fuzz.partial_ratio(main_term, name) / 100 * 4.0

    for tok in tokens:
        if tok in name:  score += 2.0
        elif tok in full: score += 0.5

    for color in colors:
        if color in name or color in vcols: score += 6.0
        elif color in full:                 score += 1.5

    score += min(p.get("likeCount", 0),  50) * 0.02
    score += min(p.get("saveCount", 0),  50) * 0.02
    score += min(p.get("viewCount", 0), 200) * 0.005

    if p.get("status") == "active":                               score += 1.0
    if (p.get("totalStock") or 0) > 0 and not p.get("isSold"):   score += 1.0

    return score


def search_products(db, tokens: list, colors: set, main_term: str, price_limit) -> list:
    q = build_product_query(tokens, colors, main_term)
    if not q:
        return []

    raw = list(db.products.find(q))  # ALL matching — no limit

    # Price filter
    if price_limit is not None:
        raw = [p for p in raw
               if (p.get("discountedPrice") or p.get("price") or 0) <= price_limit]

    # Score & sort
    scored = sorted(
        ((score_product(p, tokens, colors, main_term), p) for p in raw),
        key=lambda x: x[0], reverse=True
    )

    results = []
    for score, p in scored:
        price = p.get("discountedPrice") or p.get("price") or 0
        results.append({
            "type":  "product",
            "id":    oid_str(p.get("_id")),
            "name":  p.get("name", ""),
            "price": price,
        })

    return results


# ─────────────────────────────────────────────
# SEMANTIC QUERY EMBEDDING (with hard time budget)
# ─────────────────────────────────────────────

def get_query_vector(text: str):
    """
    Embed the query with a hard time budget. Returns None on any failure
    (timeout, API error, missing key) so the caller can fall back to
    keyword-only search — search must never 500 because Voyage hiccupped.
    """
    if not text:
        return None
    try:
        future = _embed_executor.submit(embed_query, text)
        return future.result(timeout=QUERY_EMBED_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        logger.warning("Query embedding timed out after %.1fs", QUERY_EMBED_TIMEOUT_SECONDS)
        return None
    except Exception as e:
        logger.warning("Query embedding failed: %s", e)
        return None


# ─────────────────────────────────────────────
# MAIN SEARCH
# ─────────────────────────────────────────────

def do_search(raw_query: str) -> dict:
    db = get_db()

    corrected   = correct_query(raw_query)
    price_limit = extract_price_limit(corrected)
    colors      = extract_colors(corrected)
    tokens      = search_tokens(corrected)
    main_term   = tokens[0] if tokens else ""

    if not tokens:
        return {"success": False, "message": "No products found",
                "products": [], "sellers": []}

    # ── Keyword leg (existing regex/fuzzy search, unchanged) ───
    keyword_sellers  = search_sellers(db, tokens)
    keyword_products = search_products(db, tokens, colors, main_term, price_limit)

    # ── Semantic leg (meaning-based, via Voyage + $vectorSearch) ─
    # Embed the token string (price/stop words already stripped by
    # search_tokens), not the raw query — e.g. "wedding outfit", not
    # "wedding outfit under 3000".
    query_vec = get_query_vector(" ".join(tokens))
    search_mode = "keyword-only"
    semantic_prods, semantic_sells = [], []
    if query_vec:
        semantic_prods = semantic_products(db, query_vec, price_limit=price_limit)
        semantic_sells = semantic_sellers(db, query_vec)
        search_mode = "hybrid"

    # ── Merge both legs (Reciprocal Rank Fusion) ────────────────
    products = rrf_merge(keyword_products, semantic_prods)
    sellers  = rrf_merge(keyword_sellers, semantic_sells)

    # ── Decide what to return ──────────────────
    # If sellers found AND no strong product match → seller-only result.
    # This prevents "mukura" returning sarees just because mukura sells
    # sarees. Semantic/hybrid matches are kept even without a literal
    # name match, since that's the whole point of semantic search.
    if sellers and products:
        direct_product_matches = [
            p for p in products
            if any(tok in normalize(p.get("name", "")) for tok in tokens)
            or p.get("matchType") in ("semantic", "hybrid")
        ]
        if not direct_product_matches:
            # The products came from seller lookup only — don't show them
            products = []

    total = len(products) + len(sellers)
    if total == 0:
        return {
            "success": False,
            "message": "No products found",
            "query": raw_query,
            "correctedQuery": corrected,
            "searchMode": search_mode,
            "products": [],
            "sellers": [],
        }

    return {
        "success":        True,
        "query":          raw_query,
        "correctedQuery": corrected,
        "priceFilter":    price_limit,
        "colorsDetected": list(colors),
        "searchMode":     search_mode,   # "hybrid" | "keyword-only"
        "totalProducts":  len(products),
        "totalSellers":   len(sellers),
        "products":       products,   # each: {type, id, name, price, matchType}
        "sellers":        sellers,    # each: {type, id, username, shopName, profilePic, followers, matchType}
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get("/search")
def search(query: str = ""):
    if not query.strip():
        return {"success": False, "message": "Query is empty",
                "products": [], "sellers": []}
    return do_search(query)