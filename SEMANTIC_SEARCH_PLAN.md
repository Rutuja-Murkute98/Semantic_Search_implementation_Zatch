# Semantic Search Plan — Zatch Search API

> Goal: upgrade search so it understands **meaning**, not just letters.
> Today `"wedding outfit"` returns nothing unless a product literally contains
> the word "wedding". After this plan, it will return lehengas, sarees, and
> sherwanis — because the system knows they *mean* wedding wear.

---

## 1. Where we are today

```
User query ──► typo fix (CORRECTIONS dict) ──► regex $or over
              price/color extraction          name/tags/category
                                              in MongoDB
                                                    │
                                              score in Python ──► JSON
```

**What works:** exact/partial word matches, typo fixes for known words,
price + color filters.

**What fails (and why we need semantic search):**

| Query | Today | Why it fails |
|---|---|---|
| `wedding outfit` | ❌ nothing | no product has the literal word "outfit" |
| `something for office wear` | ❌ nothing | meaning-based, not word-based |
| `ethnic dress for diwali` | ❌ poor | "ethnic", "diwali" not in product text |
| `gift for wife` | ❌ nothing | fully semantic |
| `shop that sells sarees` | ❌ wrong | seller search only matches shop *names* |
| `sadi` (Hinglish for saree) | ❌ unless in typo dict | dictionary can never cover everything |

The typo dictionary (`CORRECTIONS`, ~60 entries) is a losing battle —
semantic search makes most of it unnecessary because similar words land
close together in vector space anyway.

---

## 2. Semantic search in 30 seconds (the idea)

1. An **embedding model** turns any text into a list of numbers (a **vector**),
   e.g. 1024 numbers. Texts with similar *meaning* get similar vectors.
2. We pre-compute a vector for **every product** and **every seller** and
   store it in MongoDB next to the document.
3. At query time we turn the **user's query** into a vector too, and ask
   MongoDB: *"which stored vectors are closest to this one?"*
   Closest = most similar in meaning.

```
"wedding outfit"  ──embed──►  [0.12, -0.83, ...]
                                     │  nearest neighbours
                                     ▼
   lehenga (0.91 similar)  saree (0.88)  sherwani (0.85)  jeans (0.21)
```

---

## 3. Key decisions (recommended path)

| Decision | Recommendation | Why |
|---|---|---|
| **Where vectors live** | MongoDB **Atlas Vector Search** (`$vectorSearch`) | We already use MongoDB; no new database. Works on free M0 tier (up to 3 search indexes). |
| **Embedding model** | **Voyage AI** API — `voyage-3.5-lite` | Owned by MongoDB, generous free tier (200M tokens), great multilingual quality (handles Hinglish like "sadi"), just an HTTP call — nothing heavy runs on our small Render instance. |
| **Search style** | **Hybrid** = semantic + existing keyword search, merged | Semantic finds meaning; keyword is still best for exact names ("mukura", "iPhone 13"). Merging both gives the best of both worlds. |
| **Price / color filters** | Keep exactly as-is | These are structured filters, not a semantic problem. Applied on top of results. |

**Fallbacks if constraints change:**
- *Not on Atlas?* → Plan B: store vectors in Mongo anyway, compute cosine
  similarity in Python with numpy. Fine while catalog < ~20k products
  (the current code already loads all matches into Python, so this is no worse).
- *No external API wanted?* → Plan C: `fastembed` library (small ONNX model,
  ~100 MB RAM, runs locally, no API key). Slightly lower quality.

---

## 4. Target architecture

```
                        GET /search?query=wedding outfit under 3000
                                        │
                                        ▼
                        ┌──────────────────────────────┐
                        │ 1. normalize + extract        │
                        │    price limit & colors       │  (existing code, kept)
                        └──────────────┬───────────────┘
                                       │
                     ┌─────────────────┴──────────────────┐
                     ▼                                    ▼
        ┌────────────────────────┐          ┌────────────────────────┐
        │ 2a. SEMANTIC leg       │          │ 2b. KEYWORD leg        │
        │ embed query (Voyage)   │          │ existing regex search  │
        │ $vectorSearch on       │          │ (search.py, unchanged) │
        │ products + sellers     │          │                        │
        └───────────┬────────────┘          └───────────┬────────────┘
                    │                                   │
                    └────────────┬──────────────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ 3. MERGE (Reciprocal Rank    │
                  │    Fusion) + price/color     │
                  │    filter + dedupe           │
                  └──────────────┬───────────────┘
                                 ▼
                        same JSON response as today
                        (backward compatible)


   OFFLINE (separate from request path):
        ┌────────────────────────────────────────────┐
        │ scripts/backfill_embeddings.py             │
        │  • finds products/sellers with no vector   │
        │    or with changed text                    │
        │  • calls Voyage in batches                 │
        │  • writes `embedding` field back to Mongo  │
        │  → run once for backfill, then on a        │
        │    schedule (Render cron) to stay fresh    │
        └────────────────────────────────────────────┘
```

---

## 5. What text do we embed?

The embedding is only as good as the text we feed it. Build one
"search document" string per record:

**Product** (`products` collection → field `embedding`):
```
name. category > subCategory. tags. searchKeywords.
colors: <variant colors>. description (first ~500 chars).
```
Example: `"Banarasi Silk Saree. Women > Sarees. wedding, festive, silk.
colors: red, maroon. Handwoven Banarasi saree with golden zari border..."`

**Seller** (`users` collection → field `embedding`):
```
username. shopName. businessName. categoryType.
sells: <top categories of their products>. bio/description if present.
```
The `sells:` part is the magic — it lets `"shop that sells sarees"`
find a seller even if "saree" isn't in the shop name.

Also store `embeddingHash` (md5 of the text we embedded) so the backfill
script can detect when a document changed and needs re-embedding.

---

## 6. Implementation phases

Do these in order — each phase is small, testable, and ships value on its own.

### Phase 0 — Prerequisites (½ day)
- [ ] Confirm MongoDB is on **Atlas** (needed for `$vectorSearch`). If not → Plan B.
- [ ] Create a **Voyage AI** account, get `VOYAGE_API_KEY` (free tier).
- [ ] Add env vars: `VOYAGE_API_KEY` locally and in Render dashboard.
- [ ] Fix `requirements.txt` encoding (it's UTF-16; save as UTF-8) and add:
      `voyageai`, `numpy`.

### Phase 1 — Embedding module (1 day)
New file **`app/embeddings.py`**:
```python
def embed_texts(texts: list[str], input_type: str) -> list[list[float]]:
    """input_type: 'document' for products/sellers, 'query' for user queries."""
    # voyage client, model="voyage-3.5-lite", batching, retry on failure

def product_to_text(p: dict) -> str: ...   # builds the string from §5
def seller_to_text(s: dict, top_categories: list[str]) -> str: ...
```
- [ ] Unit-test `product_to_text` / `seller_to_text` with sample docs.
- [ ] Test one real API call manually.

### Phase 2 — Backfill script (1 day)
New file **`scripts/backfill_embeddings.py`**:
- [ ] Loop products & sellers in batches of ~100.
- [ ] Skip docs whose `embeddingHash` matches current text (cheap re-runs).
- [ ] Write `embedding` (vector) + `embeddingHash` back with `update_one`.
- [ ] Run it, verify vectors exist: `db.products.count_documents({"embedding": {"$exists": true}})`.
- [ ] Schedule it (Render Cron Job, e.g. every 6 h) so new/edited products
      get vectors automatically. *(Later improvement: embed on product-create
      in the main backend, but the cron keeps this service decoupled.)*

### Phase 3 — Vector indexes in Atlas (½ day)
In Atlas UI (or via `create_search_index`), create two indexes:

```jsonc
// index: product_vector_index  (collection: products)
{
  "fields": [
    { "type": "vector", "path": "embedding",
      "numDimensions": 1024, "similarity": "cosine" },
    { "type": "filter", "path": "status" },          // pre-filter: active only
    { "type": "filter", "path": "discountedPrice" }  // pre-filter: price
  ]
}
// index: seller_vector_index  (collection: users) — same shape, vector only
```
- [ ] Verify with a hand-written `$vectorSearch` aggregation in Atlas shell.

### Phase 4 — Semantic search leg (1–2 days)
New file **`app/vector_search.py`**:
```python
def semantic_products(db, query_vec, price_limit=None, limit=30) -> list[dict]:
    # $vectorSearch pipeline, numCandidates=200, filter on status/price
    # returns docs + vectorSearchScore

def semantic_sellers(db, query_vec, limit=10) -> list[dict]: ...
```
Notes:
- `numCandidates` ≈ 10× `limit` is a good starting point.
- Price filter goes **inside** `$vectorSearch.filter` (fast) — keep the
  Python-side filter too as a safety net for docs missing the field.

### Phase 5 — Hybrid merge & routing (1–2 days)
New file **`app/hybrid.py`**:
```python
def rrf_merge(keyword_results, semantic_results, k=60) -> list[dict]:
    # Reciprocal Rank Fusion: score(doc) = Σ 1/(k + rank_in_each_list)
    # simple, no tuning of raw scores needed, dedupe by id
```
Changes to **`app/search.py`** (`do_search`):
1. Keep steps: correct → price → colors → tokens (unchanged).
2. Embed the query **once** (strip price words first: embed `"wedding outfit"`,
   not `"wedding outfit under 3000"`).
3. Run keyword leg + semantic leg (products and sellers).
4. `rrf_merge` each pair; keep the existing "seller-name vs product" routing
   heuristic on the merged lists.
5. Same response JSON as today + two new optional fields:
   `"searchMode": "hybrid" | "keyword-only"`, and per-item `"matchType"`.
6. **Graceful degradation:** if Voyage call fails or times out (~2 s budget),
   log it and return keyword-only results. Search must never 500 because
   an embedding API hiccupped.

### Phase 6 — Evaluate, tune, clean up (ongoing)
- [ ] Build a tiny golden-query file `tests/golden_queries.json`:
      ~30 real queries → expected product/seller ids
      (include the failing examples from §1).
- [ ] Script that runs them and reports hit-rate before/after.
- [ ] Tune: RRF `k`, semantic `limit`, `numCandidates`, min similarity cutoff
      (drop results with vectorSearchScore < ~0.6 to avoid nonsense matches).
- [ ] Shrink `CORRECTIONS` to only proven cases (semantic handles most typos).
- [ ] Delete dead code: unused `is_seller_query`, duplicate dict keys.

---

## 7. Final file layout

```
main.py                      (unchanged)
render.yaml                  (+ cron job for backfill)
requirements.txt             (UTF-8; + voyageai, numpy)
app/
  __init__.py
  db.py                      (unchanged)
  search.py                  (keyword leg + orchestration in do_search)
  embeddings.py              (NEW — Voyage client + text builders)
  vector_search.py           (NEW — $vectorSearch queries)
  hybrid.py                  (NEW — RRF merge)
scripts/
  backfill_embeddings.py     (NEW)
tests/
  golden_queries.json        (NEW)
  run_eval.py                (NEW)
```

---

## 8. Costs & limits (why this stays cheap)

- **Voyage free tier:** 200M tokens. A product doc ≈ 100 tokens →
  embedding **2 million products costs $0**. Queries ≈ 5 tokens each →
  40M free searches. Paid rate after that is ~$0.02 / 1M tokens (lite model).
- **Atlas M0 (free):** supports Vector Search, max 3 search indexes
  (we need 2), 512 MB storage. 1024-dim vector ≈ 4 KB/doc → ~50k products
  fit comfortably. Upgrade tier only when catalog outgrows this.
- **Render:** no extra RAM needed — embedding happens via API, not in-process.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Voyage API down/slow | 2 s timeout → fall back to keyword-only (Phase 5.6) |
| Not actually on Atlas | Plan B: numpy cosine similarity in Python (works < ~20k docs) |
| Stale vectors after product edits | `embeddingHash` + scheduled backfill (Phase 2) |
| Weird semantic matches | similarity cutoff + hybrid keeps exact keyword matches on top |
| Query latency grows | semantic leg adds ~150–300 ms (1 API call + indexed vector search) — acceptable; cache query embeddings for popular queries later if needed |

---

## 10. Glossary (plain words)

- **Embedding / vector** — a list of numbers representing the *meaning* of text.
- **Cosine similarity** — how close two vectors point in the same direction; 1 = same meaning, 0 = unrelated.
- **`$vectorSearch`** — MongoDB Atlas's "find nearest vectors" query.
- **Hybrid search** — running keyword and semantic search together and merging.
- **RRF (Reciprocal Rank Fusion)** — a merge trick that only uses each result's *rank* in each list, so we never have to compare incompatible scores.
- **Backfill** — one-time job that computes embeddings for everything already in the database.
