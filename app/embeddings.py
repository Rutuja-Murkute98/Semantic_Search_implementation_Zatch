"""
embeddings.py — Voyage AI embedding client + text builders for semantic search.

Two things live here:
  1. embed_texts()/embed_query() — talk to Voyage AI, with batching + retry.
  2. product_to_text()/seller_to_text() — turn a Mongo document into the
     plain-text "search document" that gets embedded (see plan §5).
"""
import hashlib
import logging
import os
import time

import voyageai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("embeddings")

MODEL = "voyage-3.5-lite"
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

# Voyage accounts without a billing method on file are capped at 3
# requests/minute. We don't assume this — we only start self-pacing after
# actually seeing a rate-limit response, so a paid account (no limit) never
# gets artificially throttled. This state is process-wide (module-level) so
# it applies across separate embed_texts() calls, e.g. the many calls a
# backfill run makes, not just batches within a single call.
RATE_LIMIT_PACE_SECONDS = 21
_rate_limited = False
_last_call_time = 0.0

_client = None


def get_client() -> voyageai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set in environment")
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed_texts(texts: list, input_type: str = "document", max_retries: int = None) -> list:
    """
    Embed a list of texts with Voyage AI.

    input_type: "document" for products/sellers being indexed,
                "query" for user search queries.
    max_retries: defaults to MAX_RETRIES. Pass 1 for time-boxed callers
                 (e.g. a live search query with a hard 2s budget) where
                 retrying — let alone rate-limit pacing, which waits up to
                 RATE_LIMIT_PACE_SECONDS — could never finish in time
                 anyway, so it would just tie up a thread pointlessly.

    Batches requests (BATCH_SIZE at a time) and retries transient failures
    with backoff. Returns vectors in the same order as `texts`.

    Raises the last exception if a batch exhausts its retries — callers
    (e.g. the live search path) must catch this and fall back to
    keyword-only search rather than letting a request 500.
    """
    if not texts:
        return []

    global _rate_limited, _last_call_time

    retries = max_retries if max_retries is not None else MAX_RETRIES
    client = get_client()
    vectors = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        last_err = None
        for attempt in range(1, retries + 1):
            if _rate_limited and retries > 1:
                elapsed = time.time() - _last_call_time
                if elapsed < RATE_LIMIT_PACE_SECONDS:
                    time.sleep(RATE_LIMIT_PACE_SECONDS - elapsed)
            try:
                result = client.embed(batch, model=MODEL, input_type=input_type)
                _last_call_time = time.time()
                vectors.extend(result.embeddings)
                last_err = None
                break
            except Exception as e:
                last_err = e
                _last_call_time = time.time()
                if "RateLimitError" in type(e).__name__:
                    _rate_limited = True
                logger.warning(
                    "Voyage embed batch %d attempt %d/%d failed: %s",
                    i // BATCH_SIZE, attempt, retries, e,
                )
                if attempt < retries:
                    wait = RATE_LIMIT_PACE_SECONDS if _rate_limited else RETRY_BACKOFF_SECONDS * attempt
                    time.sleep(wait)
        if last_err is not None:
            raise last_err

    return vectors


def embed_query(text: str):
    """Embed a single query string. Returns a vector (list[float])."""
    if not text or not text.strip():
        return None
    vectors = embed_texts([text], input_type="query", max_retries=1)
    return vectors[0] if vectors else None


def text_hash(text: str) -> str:
    """md5 of the embedded text, so the backfill script can skip unchanged docs."""
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def product_to_text(p: dict) -> str:
    """Builds the search-document string embedded for a product (plan §5)."""
    name = p.get("name", "") or ""
    category = p.get("category", "") or ""
    sub_category = p.get("subCategory", "") or ""
    tags = ", ".join(p.get("tags") or [])
    keywords = ", ".join(p.get("searchKeywords") or [])
    colors = ", ".join(
        v.get("color", "") for v in (p.get("variants") or []) if v.get("color")
    )
    description = (p.get("description") or "")[:500]

    category_line = " > ".join(part for part in (category, sub_category) if part)

    parts = [name, category_line, tags, keywords]
    if colors:
        parts.append(f"colors: {colors}")
    if description:
        parts.append(description)

    return ". ".join(part for part in parts if part).strip()


def seller_to_text(s: dict, top_categories: list = None) -> str:
    """Builds the search-document string embedded for a seller (plan §5)."""
    top_categories = top_categories or []
    sp = s.get("sellerProfile") or {}

    username = s.get("username", "") or ""
    shop_name = sp.get("shopName", "") or ""
    business_name = sp.get("businessName", "") or ""
    category_type = s.get("categoryType", "") or ""
    bio = sp.get("bio") or sp.get("description") or ""

    parts = [username, shop_name, business_name, category_type]
    if top_categories:
        parts.append(f"sells: {', '.join(top_categories)}")
    if bio:
        parts.append(bio)

    return ". ".join(part for part in parts if part).strip()
