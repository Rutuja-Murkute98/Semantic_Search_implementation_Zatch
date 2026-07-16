# Semantic Search — Project Status

> Companion to [SEMANTIC_SEARCH_PLAN.md](SEMANTIC_SEARCH_PLAN.md) (the design)
> this file tracks the *implementation* — what's built, what's verified,
> what's blocked, and the exact steps to unblock it.
>
> Last updated: 2026-07-09

---

## 1. One-paragraph summary

All code for hybrid (keyword + semantic) search is written, verified, and
confirmed connecting to your real Voyage AI account and real Atlas cluster.
**Nothing is deployed to production yet.** The only thing stopping semantic
search from working *right now* is a database permission: the credential
in `MONGO_URI` is read-only, so the one-time script that writes embeddings
onto your products/sellers cannot save its output. That needs your Atlas
org owner (Rakshit) to change one user's role. Everything else — including
production deploy — is a checklist away, no further design work needed.

---

## 2. Status at a glance

| # | Step | Status | Blocked by |
|---|---|---|---|
| 1 | Code: embedding module, vector search, RRF hybrid merge, graceful fallback | ✅ Done | — |
| 2 | Code: backfill script, index-creation script, eval harness, unit tests | ✅ Done | — |
| 3 | `requirements.txt`, `render.yaml`, `.env.example` updated | ✅ Done | — |
| 4 | Local verification (unit tests, mongomock integration, real API calls) | ✅ Done | — |
| 5 | Real Mongo Atlas connection confirmed (141 products, 246 users read) | ✅ Done | — |
| 6 | Real Voyage AI connection confirmed (live embedding call returned a 1024-dim vector) | ✅ Done | — |
| 7 | **Run backfill to write embeddings onto real products/sellers** | ❌ Blocked | `rutujazatch_db_user` is read-only — needs `readWrite` on `zatch` database |
| 8 | Create the 2 Atlas Vector Search indexes | ⚠️ Blocked (not urgent) | Current Atlas login lacks project permission to create Search indexes |
| 9 | Live end-to-end test against real catalog data | ⏸ Waiting on #7 | — |
| 10 | Tune (RRF `k`, similarity cutoff) using golden queries | ⏸ Waiting on #9 | — |
| 11 | Deploy to Render (web + cron service) | ⏸ Waiting on #7 | — |

Legend: ✅ done and verified · ❌ hard blocker, nothing to do locally until fixed · ⚠️ soft blocker, has a working fallback · ⏸ next in line, waiting on an earlier step

---

## 3. What's completed (and how it was verified)

### 3.1 Code — all six phases of the plan

| File | Purpose |
|---|---|
| [app/embeddings.py](app/embeddings.py) | Voyage AI client (`voyage-3.5-lite`), batched `embed_texts()`/`embed_query()` with retry+backoff, `product_to_text()` / `seller_to_text()` document builders, `text_hash()` for change detection |
| [app/vector_search.py](app/vector_search.py) | `semantic_products()` / `semantic_sellers()` — runs Atlas `$vectorSearch`; **automatically falls back to in-process numpy cosine similarity** if the index doesn't exist yet or the cluster isn't on Atlas |
| [app/hybrid.py](app/hybrid.py) | `rrf_merge()` — Reciprocal Rank Fusion merge of the keyword and semantic result lists, tags each result `keyword` / `semantic` / `hybrid` |
| [app/search.py](app/search.py) | `do_search()` now runs both legs, embeds the query with a **hard 2-second timeout**, merges via RRF. Falls back to keyword-only silently on any embedding failure — search can never 500 because Voyage is slow or down |
| [scripts/backfill_embeddings.py](scripts/backfill_embeddings.py) | One-time + repeatable job: computes embeddings for all products/sellers, skips unchanged docs via `embeddingHash` |
| [scripts/create_vector_indexes.py](scripts/create_vector_indexes.py) | Creates the two Atlas Vector Search indexes via pymongo |
| [tests/test_embeddings.py](tests/test_embeddings.py) | 4 unit tests on the text builders — passing |
| [tests/golden_queries.json](tests/golden_queries.json) + [tests/run_eval.py](tests/run_eval.py) | Hit-rate eval harness for tuning (Phase 6) |

### 3.2 Infra / config

- `requirements.txt` — fixed from UTF-16 to UTF-8, added `voyageai`, `numpy` (versions adjusted from the plan's suggestions to ones that actually install: `voyageai==0.4.1`, `numpy==2.1.3` — the plan's `0.2.5`/`1.26.4` don't have installable/compatible builds)
- `render.yaml` — added `VOYAGE_API_KEY` to the web service, added a `zatch-search-backfill` cron service (every 6h)
- `.env.example` / `.env` — created; `.env` is gitignored
- Removed dead code flagged in the plan's Phase 6: unused `is_seller_query()`, duplicate `CORRECTIONS` dict keys

### 3.3 Verification performed

1. **Static**: every file byte-compiles (`py_compile`), no syntax errors
2. **Unit**: `tests/test_embeddings.py` — 4/4 passing
3. **Integration (mocked)**: a mongomock-backed script proved three scenarios work correctly —
   - No API key configured → `searchMode: "keyword-only"`, correct keyword results, no crash
   - Embedding call raises mid-request (simulated Voyage outage) → falls back to keyword-only, no crash, no 500
   - Embedding succeeds → `searchMode: "hybrid"`, semantic leg correctly surfaces a product via the automatic cosine-similarity fallback (proving Plan B works, since mongomock doesn't support `$vectorSearch` — the same situation as an Atlas cluster with the index not yet created)
4. **Real credentials**:
   - `get_db()` connected to your actual Atlas cluster — read 141 products, 246 users
   - `embed_query("wedding outfit")` made a live call to Voyage AI and returned a real 1024-dimension vector

Nothing here is theoretical — the code path has been exercised against your real Mongo and real Voyage account, just not yet allowed to *write* anything.

---

## 4. What's remaining, why, and how to finish it

### 4.1 Grant write access to the search service's DB user — BLOCKING

**What's blocked:** `scripts/backfill_embeddings.py` failed with:
```
OperationFailure: user is not allowed to do action [update] on [zatch.products]
```

**Why:** `.env`'s `MONGO_URI` authenticates as `rutujazatch_db_user`, whose Atlas
role is `readAnyDatabase` (read-only). The backfill needs to `update_one()`
each product/seller document to save its `embedding` field — that's a
write, which this user isn't permitted to do. This is a deliberate Atlas
permission, not a code bug; the fix has to happen in Atlas, not in code.

**What's required:** Someone with Atlas org/project admin rights (Rakshit,
per the Database Users screenshot) needs to open **Database Access** →
edit `rutujazatch_db_user` → change its role from `readAnyDatabase` to
`readWrite` (scoped to the `zatch` database) or `readWriteAnyDatabase`.
Takes effect in under a minute, no restart required.

**Step by step:**
1. Log into Atlas with an account that has Project Owner (or Database
   Access Admin) permissions
2. Project 0 → Database Access → find `rutujazatch_db_user` → **Edit**
3. Under Built-in Role, select **Read and write to any database**
   (or, for tighter scope: Specific Privileges → `readWrite` on `zatch`)
4. **Update User**
5. Tell me it's done — I'll re-run `scripts/backfill_embeddings.py`
   immediately; for 141 products + a couple hundred sellers this should
   take well under a minute and cost effectively $0 of Voyage's free tier

**Alternative unblock (if this route is slow):** if anyone can share
credentials for the existing `zatch-app` user (already has
`readWriteAnyDatabase`), swapping `MONGO_URI` to use it works just as well
and needs no Atlas changes at all.

### 4.2 Create the two Atlas Vector Search indexes — NOT BLOCKING, but recommended

**What's blocked:** `scripts/create_vector_indexes.py` failed with:
```
OperationFailure: user is not allowed to do action [createSearchIndexes] on [zatch.products]
```

**Why:** creating Atlas Search/Vector indexes is gated by the *human*
Atlas login's project role (Project Owner / Cluster Manager), separate
from database user roles entirely. The account currently used to browse
Atlas doesn't have that project permission.

**Why this doesn't block going live:** `app/vector_search.py` was built
to detect exactly this situation — if `$vectorSearch` fails, it
automatically computes cosine similarity in Python over documents that
have an `embedding` field (capped at 20,000 docs; you have 141 products).
This is slightly slower per-query than Atlas's native index but returns
**identical, correct results**. So semantic search works end-to-end the
moment step 4.1 is resolved, with or without these indexes.

**What's required (whenever convenient):** either
- (a) grant your Atlas login Project Owner / Cluster Manager role, or
- (b) have Rakshit create the two indexes directly

**Step by step (for whoever has access), via Atlas UI → your cluster →
Atlas Search tab → Create Search Index → JSON Editor:**

Index 1 — database `zatch`, collection `products`, name `product_vector_index`:
```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine" },
    { "type": "filter", "path": "status" },
    { "type": "filter", "path": "discountedPrice" }
  ]
}
```

Index 2 — database `zatch`, collection `users`, name `seller_vector_index`:
```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine" }
  ]
}
```

Both take ~1 minute to finish building. Your M0 cluster has a 3-index
limit and is currently using 1 (a pre-existing text index on `products`),
so there's room for both.

Once these exist, `app/vector_search.py` will start using them
automatically on the next process restart — no code or config change
needed, and no need to tell me unless you want it confirmed.

### 4.3 Run the backfill against real data — waiting on 4.1

Once write access is granted:
```
python scripts/backfill_embeddings.py
```
This computes and saves an `embedding` vector (+ `embeddingHash`) for
every product and seller. Expected runtime: seconds, given the current
catalog size (141 products, 246 users — though not all users are sellers,
so likely fewer). I'll run this and report the counts once unblocked.

### 4.4 Live end-to-end test against real catalog — waiting on 4.3

Run a handful of the "today fails" queries from the plan's §1 table
(`"wedding outfit"`, `"gift for wife"`, `"shop that sells sarees"`, etc.)
through `do_search()` and eyeball whether the results make sense for your
actual product catalog. This is the first point where we find out how
well the model performs on *your* specific inventory and product-text
quality — some tuning is normal and expected here.

### 4.5 Tune with the golden-query harness — waiting on 4.4

`tests/golden_queries.json` currently has placeholder queries with empty
`expectedIds`. Once step 4.4 shows real product/seller IDs that *should*
match each query, fill those in, then run:
```
python tests/run_eval.py
```
Use the hit-rate to tune (in `app/vector_search.py` / `app/hybrid.py`):
- `SIM_CUTOFF` (currently 0.6) — raise if semantic matches look like noise, lower if good matches are getting dropped
- RRF `k` (currently 60) — lower values weight top-ranked results more heavily
- `numCandidates` in the `$vectorSearch` pipeline — raise for better recall at a latency cost

### 4.6 Deploy — waiting on 4.1 (and ideally 4.4)

`render.yaml` is already updated with a `VOYAGE_API_KEY` env var slot on
the web service and a new `zatch-search-backfill` cron service. To go
live on Render:
1. In the Render dashboard, set `MONGO_URI` and `VOYAGE_API_KEY` on both
   the `zatch-search` web service and the `zatch-search-backfill` cron job
2. Push/deploy — Render will run `pip install -r requirements.txt` and
   start both services from the updated `render.yaml`
3. Confirm the cron job's first run succeeds (check Render logs) so
   embeddings stay fresh as products change

### 4.7 Ongoing cleanup (optional, low priority)

Per the plan's Phase 6: once semantic search is live and proven, shrink
`CORRECTIONS` in `app/search.py` to only the typo cases that semantic
search still doesn't catch — most of that ~60-entry dictionary becomes
redundant once similar words land close together in vector space.

---

## 5. What you need to provide / arrange (summary)

| Need | From whom | For which step |
|---|---|---|
| `rutujazatch_db_user` role changed to `readWrite`/`readWriteAnyDatabase` (or working credentials for a user that already has it) | Rakshit / Atlas org admin | 4.1 — unblocks everything downstream |
| Project Owner/Cluster Manager role on your Atlas login, **or** Rakshit creates the 2 indexes using the JSON above | Rakshit / Atlas org admin | 4.2 — optional, improves performance at scale |
| A few real product/seller IDs that should match specific test queries | You, after eyeballing 4.4's results | 4.5 — tuning |
| `MONGO_URI` + `VOYAGE_API_KEY` set in Render dashboard | You | 4.6 — deploy |

---

## 6. Current file layout

```
main.py                          unchanged
render.yaml                      + VOYAGE_API_KEY, + backfill cron service
requirements.txt                 fixed to UTF-8; + voyageai, numpy
.env.example / .env              new (credentials; .env gitignored)
app/
  __init__.py
  db.py                          unchanged
  search.py                      keyword leg + hybrid orchestration (v5)
  embeddings.py                  NEW — Voyage client + text builders
  vector_search.py               NEW — $vectorSearch + automatic cosine fallback
  hybrid.py                      NEW — RRF merge
scripts/
  backfill_embeddings.py         NEW — writes embeddings to Mongo
  create_vector_indexes.py       NEW — one-time Atlas index setup
tests/
  test_embeddings.py             NEW — unit tests (passing)
  golden_queries.json            NEW — needs real expectedIds filled in
  run_eval.py                    NEW — hit-rate eval harness
```

---

## 7. Quick reference — next action

The single next action, for whoever has Atlas admin rights, is:

> **Change `rutujazatch_db_user`'s role to `readWrite` on the `zatch`
> database** (Atlas → Database Access → Edit).

Everything else in this document is either already done or waiting
directly on that one change.
