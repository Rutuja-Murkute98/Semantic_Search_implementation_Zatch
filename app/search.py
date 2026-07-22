"""
search.py — Zatch Search v6 (hybrid semantic + fuzzy keyword)
=================================================================
- Keyword leg: MongoDB Atlas Search fuzzy text matching (app/text_search.py)
  handles typos natively — no hardcoded typo dictionary to maintain.
  Falls back to plain regex matching if the Atlas Search index isn't
  available yet.
- Semantic leg: embeds the query (Voyage AI) and runs $vectorSearch on
  products, sellers, and bits, so meaning-based queries like "wedding
  outfit" work even when no product literally contains that word.
- Both legs are merged with Reciprocal Rank Fusion (app/hybrid.py).
- Graceful degradation: if the embedding call fails or is slow, search
  silently falls back to keyword-only — it never 500s because of that.
- Live-status filtering: only in-stock, active products and active bits
  are returned — applied as a hard filter, not just a ranking boost.
- Seller search → returns seller info only (not their products).
- Smart routing: if query matches a seller name → show seller
  if query matches product terms → show products
  if both match → show both sections separately
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from unidecode import unidecode
from fastapi import APIRouter
from app.db import get_db
from app.embeddings import embed_query
from app.vector_search import semantic_products, semantic_sellers, semantic_bits
from app.text_search import search_products, search_sellers, search_bits
from app.hybrid import rrf_merge

router = APIRouter()
logger = logging.getLogger("search")

QUERY_EMBED_TIMEOUT_SECONDS = 2.0
_embed_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed")
# Shared pool for the 6 independent keyword/semantic retrieval calls in
# do_search(), reused across requests rather than spinning up threads
# per-request. 8 workers covers all 6 concurrent calls with headroom.
_query_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="query")

COLORS = {
    # base colors
    "red", "blue", "green", "black", "white", "yellow", "pink", "purple",
    "orange", "grey", "gray", "brown", "beige", "maroon", "navy", "golden",
    "gold", "silver", "cream", "multicolor", "multi",
    # additional base colors seen in the real catalog / common in fashion
    "ash", "coffee", "tan", "lavender", "orchid", "peach", "teal",
    "turquoise", "magenta", "coral", "mint", "khaki", "ivory", "wine",
    "rust", "olive", "bronze", "mauve", "plum", "burgundy", "charcoal",
    "fuchsia", "indigo", "lilac", "salmon", "lime", "aqua", "mustard",
    # modifier words that only ever appear as part of a compound color name
    # ("sky blue", "off white", "dark blue", "midnight blue", "sea green",
    # "deep pink", "ink blue", "royal blue") -- tokens are checked
    # word-by-word, so both halves need to be recognised for pick_main_term
    # to skip the whole phrase.
    "sky", "off", "dark", "light", "midnight", "sea", "deep", "ink",
    "royal", "baby", "hot", "neon", "pastel",
}

# General search-noise words: filler, price phrasing, and connective words
# that add nothing to what's being searched for ("saree with red color" ->
# "with" and "color" are noise, not search subjects).
STOPWORDS = {
    # price phrasing
    "under", "below", "less", "than", "upto", "up", "to", "within",
    "rs", "rupees", "only", "me", "show", "find", "get", "buy", "some",
    # articles / connectives
    "a", "an", "the", "for", "with", "and", "or", "but", "of", "in",
    "on", "at", "is", "are", "my", "your", "this", "that", "these",
    "those", "please", "want", "need", "looking", "search", "i",
    # generic e-commerce noise words that match almost everything
    "color", "colour", "item", "items", "product", "products",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unidecode(text or "").lower().strip())


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
            if w not in STOPWORDS and not w.isdigit() and len(w) > 1]


def pick_main_term(tokens: list, colors: set) -> str:
    """
    Pick the token that plays the "subject" role in the query -- the actual
    product/category being searched -- rather than just whichever token
    happens to come first.

    Colour words are modifiers: they already get their own dedicated
    handling via `colors` (separate query clauses + separate scoring
    boost). If a colour is picked as the main term merely because it's
    first ("red saree" -> main_term="red"), it gets the heaviest keyword
    weighting and skews results toward anything with "red" in the name,
    regardless of category -- e.g. "red saree" surfacing unrelated red
    electronics. Skipping colour tokens when choosing the main term fixes
    that: "red saree" -> main_term="saree", "red" still boosts via colors.
    """
    for tok in tokens:
        if tok not in colors:
            return tok
    return tokens[0] if tokens else ""


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

def _dedupe_by_name(items: list, name_key: str) -> list:
    """Some sellers list visually-identical products more than once
    (same name, different _id) -- keep only the highest-ranked occurrence
    of each name so results don't look duplicated. `items` is expected to
    already be ranked (rrf_merge's output order), so "first wins"."""
    seen = set()
    deduped = []
    for item in items:
        key = normalize(item.get(name_key, ""))
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(item)
    return deduped


def do_search(raw_query: str) -> dict:
    db = get_db()

    corrected   = normalize(raw_query)
    price_limit = extract_price_limit(corrected)
    colors      = extract_colors(corrected)
    tokens      = search_tokens(corrected)
    main_term   = pick_main_term(tokens, colors)

    if not tokens:
        return {"success": False, "message": "No products found",
                "products": [], "sellers": [], "bits": []}

    # ── Run both legs concurrently instead of as 6 sequential MongoDB
    # round-trips -- each call is independent I/O, so there's no reason
    # to wait for one before starting the next. This alone cuts typical
    # latency from ~3.5s to roughly the single slowest call (~1s).
    query_text = " ".join(tokens)
    f_kw_products = _query_executor.submit(search_products, db, tokens, colors, main_term, price_limit)
    f_kw_sellers  = _query_executor.submit(search_sellers, db, tokens)
    f_kw_bits     = _query_executor.submit(search_bits, db, tokens)
    f_vec         = _query_executor.submit(get_query_vector, query_text)

    # Semantic legs need the embedding first, but that's fast (~tens of ms
    # once the model is warm) -- start them as soon as it's ready rather
    # than waiting for the keyword legs above to finish too.
    query_vec = f_vec.result()
    search_mode = "keyword-only"
    f_sem_products = f_sem_sellers = f_sem_bits = None
    if query_vec:
        f_sem_products = _query_executor.submit(semantic_products, db, query_vec, price_limit=price_limit)
        f_sem_sellers  = _query_executor.submit(semantic_sellers, db, query_vec)
        f_sem_bits     = _query_executor.submit(semantic_bits, db, query_vec)
        search_mode = "hybrid"

    keyword_products = f_kw_products.result()
    keyword_sellers  = f_kw_sellers.result()
    keyword_bits     = f_kw_bits.result()
    semantic_prods = f_sem_products.result() if f_sem_products else []
    semantic_sells = f_sem_sellers.result() if f_sem_sellers else []
    semantic_bts   = f_sem_bits.result() if f_sem_bits else []

    # ── Merge each leg pair (Reciprocal Rank Fusion) ────────────
    products = _dedupe_by_name(rrf_merge(keyword_products, semantic_prods), "name")
    sellers  = rrf_merge(keyword_sellers, semantic_sells)
    bits     = _dedupe_by_name(rrf_merge(keyword_bits, semantic_bts), "title")

    # ── Decide what to return ──────────────────
    # If sellers found AND no strong product/bit match → seller-only result.
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
            products = []

    if sellers and bits:
        direct_bit_matches = [
            b for b in bits
            if any(tok in normalize(b.get("title", "")) for tok in tokens)
            or b.get("matchType") in ("semantic", "hybrid")
        ]
        if not direct_bit_matches:
            bits = []

    total = len(products) + len(sellers) + len(bits)
    if total == 0:
        return {
            "success": False,
            "message": "No products found",
            "query": raw_query,
            "correctedQuery": corrected,
            "searchMode": search_mode,
            "products": [],
            "sellers": [],
            "bits": [],
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
        "totalBits":      len(bits),
        "products":       products,   # each: {type, id, name, price, matchType}
        "sellers":        sellers,    # each: {type, id, username, shopName, profilePic, followers, matchType}
        "bits":           bits,       # each: {type, id, title, videoUrl, thumbnailUrl, likeCount, viewCount, matchType}
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get("/search")
def search(query: str = ""):
    if not query.strip():
        return {"success": False, "message": "Query is empty",
                "products": [], "sellers": [], "bits": []}
    return do_search(query)
