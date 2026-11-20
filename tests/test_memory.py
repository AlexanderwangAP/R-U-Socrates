"""Memory layer: dedup, bounded growth, and crash-safe persistence."""

import json

from services.worker.memory import CognitionStore, NodeDatabase
from services.worker.models import CognitionItem, Node


def test_cognition_store_rejects_duplicate_content(tmp_path):
    store = CognitionStore(storage_dir=tmp_path / "cognition")

    first = store.add(CognitionItem(content="the gradient was unstable"))
    second = store.add(CognitionItem(content="  the   gradient  was unstable "))

    assert first == second, "whitespace-only difference was stored twice"
    assert len(store) == 1


def test_cognition_store_evicts_oldest_at_capacity(tmp_path):
    store = CognitionStore(storage_dir=tmp_path / "cognition", max_size=3)

    ids = [store.add(CognitionItem(content=f"insight {i}")) for i in range(6)]

    assert len(store) == 3
    assert ids[0] not in store.items
    assert ids[-1] in store.items
    # The eviction must not mint a new int id for the item it just dropped.
    assert len(store.str_to_int) == len(store.int_to_str)


def test_cognition_store_reloads_dedup_index(tmp_path):
    store = CognitionStore(storage_dir=tmp_path / "cognition")
    store.add(CognitionItem(content="remember me"))

    reloaded = CognitionStore(storage_dir=tmp_path / "cognition")
    assert reloaded.add(CognitionItem(content="remember me")) in reloaded.items
    assert len(reloaded) == 1, "dedup index was lost across a restart"


def test_node_database_saves_without_leaving_temp_files(tmp_path):
    db = NodeDatabase(storage_dir=tmp_path / "nodes")
    db.add(Node(name="c1", code="x = 1"))

    leftovers = list(tmp_path.rglob("*.tmp"))
    assert leftovers == [], f"temp files left behind: {leftovers}"

    payload = json.loads((tmp_path / "nodes" / "nodes.json").read_text(encoding="utf-8"))
    assert payload["next_id"] == 1
    assert len(payload["nodes"]) == 1


def test_node_database_round_trips_through_disk(tmp_path):
    db = NodeDatabase(storage_dir=tmp_path / "nodes")
    db.add(Node(name="c1", code="x = 1", score=0.5))
    db.add(Node(name="c2", code="x = 2", score=0.9))

    reloaded = NodeDatabase(storage_dir=tmp_path / "nodes")
    assert len(reloaded) == 2
    assert reloaded.get_best().name == "c2"
    assert reloaded.next_id == 2, "ids would collide after a restart"
