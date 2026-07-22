"""
embeddings.py — local embedding model + text builders for semantic search.

Uses fastembed (a small open-source ONNX model that runs entirely on this
machine) instead of a paid API: no API key, no rate limits, no per-request
cost. See SEMANTIC_SEARCH_PLAN.md §3 "Plan C" for the original rationale
(this project started on Voyage AI's hosted API; switched to local
embeddings after Voyage's free-tier rate limit proved too unreliable for
batch backfills).

Two things live here:
  1. embed_texts()/embed_query() — run the local embedding model.
  2. product_to_text()/seller_to_text()/bit_to_text() — turn a Mongo
     document into the plain-text "search document" that gets embedded.
"""
import hashlib
import logging

logger = logging.getLogger("embeddings")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSIONS = 384

_model = None


def get_model():
    """Lazily loads the local embedding model (downloads + caches on
    first use, then reused for the life of the process)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def embed_texts(texts: list, input_type: str = "document", **_ignored) -> list:
    """
    Embed a list of texts locally.

    input_type: "document" for products/sellers/bits being indexed,
                "query" for user search queries. BGE models are trained
                asymmetrically (queries and documents use different
                encodings for best retrieval quality), so this picks
                fastembed's query_embed() vs passage_embed() accordingly.

    Returns a list of vectors (list[list[float]]) in the same order as
    `texts`. Runs locally -- no network call, no rate limit, so there is
    no batching/retry/backoff logic to worry about here.
    """
    if not texts:
        return []

    model = get_model()
    if input_type == "query":
        vectors = model.query_embed(texts)
    else:
        vectors = model.passage_embed(texts)
    return [v.tolist() for v in vectors]


def embed_query(text: str):
    """Embed a single query string. Returns a vector (list[float])."""
    if not text or not text.strip():
        return None
    vectors = embed_texts([text], input_type="query")
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


def bit_to_text(b: dict) -> str:
    """Builds the search-document string embedded for a bit (short video post)."""
    title = b.get("title", "") or ""
    description = b.get("description", "") or ""
    hashtags = " ".join(tag.lstrip("#") for tag in (b.get("hashtags") or []))

    parts = [title]
    if description and description != title:
        parts.append(description)
    if hashtags:
        parts.append(f"tags: {hashtags}")

    return ". ".join(part for part in parts if part).strip()
