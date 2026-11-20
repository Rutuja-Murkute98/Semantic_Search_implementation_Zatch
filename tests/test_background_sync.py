"""Tests reconcile() against a tiny in-memory fake DB (no Atlas needed)."""
from app import background_sync
from app.embeddings import product_to_text, text_hash


class FakeColl:
    def __init__(self, docs=None):
        self.docs = {d["_id"]: dict(d) for d in (docs or [])}

    def find(self, flt=None, proj=None):
        flt = flt or {}
        return [dict(d) for d in self.docs.values()
                if all(d.get(k) == v for k, v in flt.items())]

    def update_one(self, flt, update, upsert=False):
        doc = self.docs.setdefault(flt["_id"], {"_id": flt["_id"]})
        doc.update(update["$set"])

    def delete_one(self, flt):
        self.docs.pop(flt["_id"], None)


def make_db(products, product_embeddings):
    return {
        "products": FakeColl(products), "product_embeddings": FakeColl(product_embeddings),
        "users": FakeColl(), "seller_embeddings": FakeColl(),
        "bits": FakeColl(), "bit_embeddings": FakeColl(),
    }


def fake_embed(monkeypatch, calls):
    def _embed(texts, input_type="document", **_):
        calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]
    monkeypatch.setattr(background_sync, "embed_texts", _embed)


def test_embeds_missing_changed_skips_unchanged_and_removes_orphans(monkeypatch):
    calls = []
    fake_embed(monkeypatch, calls)
    same = {"_id": 1, "name": "Blue Shirt"}
    changed = {"_id": 2, "name": "Red Saree"}
    missing = {"_id": 3, "name": "Green Kurta"}
    db = make_db(
        products=[same, changed, missing],
        product_embeddings=[
            {"_id": 1, "embeddingHash": text_hash(product_to_text(same))},  # up to date
            {"_id": 2, "embeddingHash": "stale"},                            # text changed
            {"_id": 99, "embeddingHash": "x"},                               # deleted product
        ],
    )
    removed = background_sync.reconcile(db)

    embedded = [t for batch in calls for t in batch]
    assert sorted(embedded) == sorted([product_to_text(changed), product_to_text(missing)])
    assert removed["products"] == 1
    assert set(db["product_embeddings"].docs) == {1, 2, 3}
    assert db["product_embeddings"].docs[3]["embedding"] == [0.1, 0.2]


def test_active_products_are_embedded_before_inactive(monkeypatch):
    calls = []
    fake_embed(monkeypatch, calls)
    db = make_db(
        products=[{"_id": 1, "name": "Draft Thing", "status": "draft"},
                  {"_id": 2, "name": "Live Thing", "status": "active"}],
        product_embeddings=[],
    )
    background_sync.reconcile(db)
    assert calls[0][0].startswith("Live Thing")


def test_never_wipes_embeddings_when_source_is_empty(monkeypatch):
    fake_embed(monkeypatch, [])
    db = make_db(products=[], product_embeddings=[{"_id": 1}, {"_id": 2}])
    removed = background_sync.reconcile(db)
    assert removed["products"] == 0
    assert set(db["product_embeddings"].docs) == {1, 2}


def test_embedding_failure_does_not_raise(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model failed")
    monkeypatch.setattr(background_sync, "embed_texts", boom)
    db = make_db(products=[{"_id": 1, "name": "X"}], product_embeddings=[])
    background_sync.reconcile(db)  # must not raise


def test_start_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKGROUND_SYNC", "0")
    assert background_sync.start() is None
