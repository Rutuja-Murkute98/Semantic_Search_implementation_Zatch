# Zatch Search API

Hybrid product/seller/bit search for Zatch — combines fuzzy keyword search
with AI-powered semantic search, so queries like *"wedding outfit"* or
*"gift for wife"* return relevant results even when no product literally
contains those words.

## How it works

1. Every product, seller, and bit (short video post) is converted into a
   "meaning vector" (embedding) by a small local, open-source model
   ([fastembed](https://github.com/qdrant/fastembed), no API key or cost)
   and stored in a **dedicated companion collection**
   (`product_embeddings`, `seller_embeddings`, `bit_embeddings`) — never
   as a field on the product/seller/bit record itself. Each companion
   document shares its `_id` with the record it describes, so the two
   are joined back together at query time via `$lookup`. This means the
   search service only ever needs write access to these three small,
   dedicated collections — never to live product/order data — and the
   AI-search layer stays fully separable from the core data model.
2. At search time, the query is embedded the same way and compared against
   stored vectors using MongoDB Atlas `$vectorSearch` on the companion
   collection, then joined back to the real product/seller/bit for
   filtering (status, price) and display fields.
3. In parallel, a keyword leg matches on structured fields (name, category,
   tags) using MongoDB Atlas Search fuzzy text matching where available —
   tolerating typos natively, no hardcoded correction dictionary needed —
   with the main search term chosen by its role in the query (skipping
   colour words, so "red saree" searches for sarees, not "red").
4. Semantic and keyword results are merged using Reciprocal Rank Fusion.
5. Only live results are returned: active, in-stock, unsold products and
   active bits — applied as a hard filter, not just a ranking boost.
6. If the embedding model or Atlas Search/Vector Search isn't available,
   search automatically falls back to plain regex matching / in-process
   cosine similarity — it never fails the request.
7. New/edited products, sellers, and bits become searchable automatically,
   within seconds — a background worker (`app/change_stream_listener.py`)
   watches the live database via MongoDB Change Streams and re-embeds only
   what changed, the moment it changes. No manual re-run, no snapshot, no
   fixed delay. A daily cron re-run of the backfill script stays in place
   purely as a safety net in case an event is ever missed (e.g. during a
   brief restart) — it's no longer the primary freshness mechanism.

## Project layout

```
main.py                    FastAPI app entrypoint
app/
  db.py                    MongoDB connection
  search.py                Keyword search + hybrid orchestration
  embeddings.py            Local embedding model + text builders
  text_search.py           Atlas Search fuzzy keyword matching + regex fallback
  vector_search.py         $vectorSearch queries + fallback
  hybrid.py                Reciprocal Rank Fusion merge
  change_stream_listener.py  Real-time embedding sync via MongoDB Change Streams
scripts/
  backfill_embeddings.py   Generates/refreshes embeddings for products/sellers/bits
  run_change_stream_listener.py  Entrypoint for the real-time sync worker
  create_vector_indexes.py One-time Atlas Vector Search index setup
  create_text_indexes.py   One-time Atlas Search fuzzy-text index setup
  export_to_staging.py     Copies real (read-only) catalog data into a staging DB
static/
  index.html               Simple search UI for testing/demos
tests/
  test_embeddings.py       Unit tests for text builders
  test_search.py           Unit tests for query-parsing logic
  test_hybrid.py           Unit tests for RRF merge + relevance-score ordering
  golden_queries.json      Sample queries for evaluating search quality
  run_eval.py              Hit-rate evaluation harness
```

## Local setup

```bash
python -m venv venv
venv/Scripts/activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # then fill in MONGO_URI
```

Run the API:

```bash
uvicorn main:app --reload
```

- `GET /search?query=...` — search endpoint
- `GET /demo/` — simple browser UI for testing search
- `GET /health` — health check

## First-time setup for semantic search

Requires write access to the `product_embeddings` / `seller_embeddings` /
`bit_embeddings` companion collections (not to `products` / `users` /
`bits` themselves — see "How it works" above) and, ideally, MongoDB
Atlas (for `$vectorSearch` / fuzzy `$search`; both fall back
automatically if unavailable).

```bash
python scripts/create_vector_indexes.py    # one-time: create Atlas vector index
python scripts/create_text_indexes.py      # one-time: create Atlas fuzzy text index
python scripts/backfill_embeddings.py      # generate embeddings for existing data
```

This one-time run covers everything that exists right now. After that,
new/edited data is picked up automatically — see below — you never need
to re-run this by hand. No API key or per-request cost is involved —
embeddings run locally.

## Keeping embeddings fresh automatically

```bash
python scripts/run_change_stream_listener.py   # long-running, never exits
```

This watches `products`, `users`, and `bits` directly via MongoDB Change
Streams and re-embeds only the document that actually changed, within
seconds — new sellers, new products, new bits, edits, and deletions are
all handled automatically. It never touches `products`/`users`/`bits`
itself (read-only there); it only writes to the 3 companion collections,
plus a small internal `_embedding_sync_state` collection that stores a
resume token per watched collection, so a restart picks up exactly where
it left off instead of missing anything.

`scripts/backfill_embeddings.py` stays in place too, now running on a
daily schedule (see `render.yaml`) purely as a safety net — it's cheap
to re-run since it skips anything whose text hasn't changed.

## Deployment

Deploys to [Render](https://render.com) using `render.yaml`: a web
service, a persistent background worker
(`zatch-search-embedding-sync`, running `run_change_stream_listener.py`
— this is what needs to stay always-on, not just spin up on a schedule),
and a daily cron job for the backfill safety net. Required environment
variable on all three services:

- `MONGO_URI`
