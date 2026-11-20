"""Tests reconcile() against a tiny in-memory fake DB (no Atlas needed)."""
from app import background_sync


class FakeColl:
    def __init__(self, docs=None):
        self.docs = {d["_id"]: dict(d) for d in (docs or [])}

    def find(self, flt=None, proj=None):
        flt = flt or {}
        return [dict(d) for d in self.docs.values()
                if all(d.get(k) == v for k, v in flt.items())]

    def delete_one(self, flt):
        self.docs.pop(flt["_id"], None)


class FakeDB(dict):
    def __getitem__(self, k):
        return dict.__getitem__(self, k)


def make_db(products, product_embeddings):
    return FakeDB(
        products=FakeColl(products), product_embeddings=FakeColl(product_embeddings),
        users=FakeColl(), seller_embeddings=FakeColl(),
        bits=FakeColl(), bit_embeddings=FakeColl(),
    )


def test_reconcile_syncs_every_source_and_removes_orphans(monkeypatch):
    synced = []
    monkeypatch.setitem(background_sync._SYNC_HANDLERS, "products",
                        lambda db, i: synced.append(i))
    db = make_db(
        products=[{"_id": 1}, {"_id": 2}],
        product_embeddings=[{"_id": 1}, {"_id": 99}],  # 99 = deleted product
    )
    removed = background_sync.reconcile(db)
    assert synced == [1, 2]                       # missing one (2) gets handled too
    assert removed["products"] == 1
    assert set(db["product_embeddings"].docs) == {1}


def test_reconcile_never_wipes_embeddings_when_source_is_empty(monkeypatch):
    db = make_db(products=[], product_embeddings=[{"_id": 1}, {"_id": 2}])
    removed = background_sync.reconcile(db)
    assert removed["products"] == 0
    assert set(db["product_embeddings"].docs) == {1, 2}


def test_reconcile_survives_a_failing_handler(monkeypatch):
    def boom(db, i):
        raise RuntimeError("embed failed")
    monkeypatch.setitem(background_sync._SYNC_HANDLERS, "products", boom)
    db = make_db(products=[{"_id": 1}], product_embeddings=[])
    background_sync.reconcile(db)  # must not raise


def test_start_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKGROUND_SYNC", "0")
    assert background_sync.start() is None
