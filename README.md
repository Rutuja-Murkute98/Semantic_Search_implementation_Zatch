# Zatch Search API

Hybrid product/seller/bit search for Zatch — combines fuzzy keyword search
with AI-powered semantic search, so queries like *"wedding outfit"* or
*"gift for wife"* return relevant results even when no product literally
contains those words.

## How it works

1. Every product, seller, and bit (short video post) is converted into a
   "meaning vector" (embedding) by a small local, open-source model
   ([fastembed](https://github.com/qdrant/fastembed), no API key or cost)
   and stored in MongoDB alongside the record.
2. At search time, the query is embedded the same way and compared against
   stored vectors using MongoDB Atlas `$vectorSearch`.
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
scripts/
  backfill_embeddings.py   Generates/refreshes embeddings for products/sellers/bits
  create_vector_indexes.py One-time Atlas Vector Search index setup
  create_text_indexes.py   One-time Atlas Search fuzzy-text index setup
  export_to_staging.py     Copies real (read-only) catalog data into a staging DB
static/
  index.html               Simple search UI for testing/demos
tests/
  test_embeddings.py       Unit tests for text builders
  test_search.py           Unit tests for query-parsing logic
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

Requires write access on the database (to store embeddings) and, ideally,
MongoDB Atlas (for `$vectorSearch` / fuzzy `$search`; both fall back
automatically if unavailable).

```bash
python scripts/create_vector_indexes.py    # one-time: create Atlas vector index
python scripts/create_text_indexes.py      # one-time: create Atlas fuzzy text index
python scripts/backfill_embeddings.py      # generate embeddings for existing data
```

Re-run the backfill any time products/sellers/bits change — it skips
documents whose text hasn't changed, so re-runs are cheap. In production
this runs on a schedule via the cron job defined in `render.yaml`. No API
key or per-request cost is involved — embeddings run locally.

## Deployment

Deploys to [Render](https://render.com) using `render.yaml` (a web service +
a cron job for the scheduled backfill). Required environment variable on
both services:

- `MONGO_URI`
