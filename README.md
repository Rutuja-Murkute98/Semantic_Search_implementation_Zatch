# Zatch Search API

Hybrid product/seller search for Zatch — combines regex/fuzzy keyword search
with AI-powered semantic search, so queries like *"wedding outfit"* or
*"gift for wife"* return relevant results even when no product literally
contains those words.

## How it works

1. Every product and seller is converted into a "meaning vector" (embedding)
   by [Voyage AI](https://www.voyageai.com/) and stored in MongoDB alongside
   the record.
2. At search time, the query is embedded the same way and compared against
   stored vectors using MongoDB Atlas `$vectorSearch`.
3. Semantic results are merged with the existing keyword search using
   Reciprocal Rank Fusion.
4. If the embedding service is ever slow or unavailable, search
   automatically falls back to keyword-only — it never fails the request.
5. If Atlas Vector Search isn't available (e.g. no index yet), the same
   fallback logic computes cosine similarity in-process instead.

## Project layout

```
main.py                    FastAPI app entrypoint
app/
  db.py                    MongoDB connection
  search.py                Keyword search + hybrid orchestration
  embeddings.py            Voyage AI client + text builders
  vector_search.py         $vectorSearch queries + fallback
  hybrid.py                Reciprocal Rank Fusion merge
scripts/
  backfill_embeddings.py   Generates/refreshes embeddings for all products/sellers
  create_vector_indexes.py One-time Atlas Vector Search index setup
  export_to_staging.py     Copies real (read-only) catalog data into a staging DB
static/
  index.html               Simple search UI for testing/demos
tests/
  test_embeddings.py       Unit tests for text builders
  golden_queries.json      Sample queries for evaluating search quality
  run_eval.py              Hit-rate evaluation harness
```

## Local setup

```bash
python -m venv venv
venv/Scripts/activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # then fill in MONGO_URI and VOYAGE_API_KEY
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
MongoDB Atlas (for `$vectorSearch`; falls back automatically if unavailable).

```bash
python scripts/create_vector_indexes.py    # one-time: create Atlas indexes
python scripts/backfill_embeddings.py      # generate embeddings for existing data
```

Re-run the backfill any time products/sellers change — it skips documents
whose text hasn't changed, so re-runs are cheap. In production this runs on
a schedule via the cron job defined in `render.yaml`.

## Deployment

Deploys to [Render](https://render.com) using `render.yaml` (a web service +
a cron job for the scheduled backfill). Required environment variables on
both services:

- `MONGO_URI`
- `VOYAGE_API_KEY`
