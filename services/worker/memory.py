"""
Memory layer for the R U Socrates worker.

Combines the node Database and Cognition store from ASI-Evolve into a single
module. Internal imports only — no HTTP, no external service.

Classes:
    EmbeddingService  — sentence-transformers wrapper
    FAISSIndex        — FAISS vector index
    NodeDatabase      — persistent node store with UCB1 sampling
    CognitionStore    — persistent cognition item store with semantic retrieval
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import Node, CognitionItem

# ---------------------------------------------------------------------------
# Optional heavy dependencies (graceful ImportError for environments without them)
# ---------------------------------------------------------------------------

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON through a temp file + rename.

    A plain overwrite left half a JSON file behind if the process died mid-write,
    which took the whole node database down with it on the next start.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


_EMBEDDING_WARNED = False


def _warn_embedding_unavailable(exc: BaseException) -> None:
    """Log the missing-embedding-stack warning once per process."""
    global _EMBEDDING_WARNED
    if _EMBEDDING_WARNED:
        return
    _EMBEDDING_WARNED = True
    logging.getLogger(__name__).warning(
        "Semantic memory disabled (%s: %s). Install faiss-cpu and "
        "sentence-transformers to enable it.",
        type(exc).__name__,
        exc,
    )


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------

class EmbeddingService:
    """Local embedding service backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension: int = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: str | List[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared embedding service
# ---------------------------------------------------------------------------

_EMBEDDING_CACHE: Dict[str, "EmbeddingService"] = {}


def get_embedding_service(model_name: str, device: str = "cpu") -> "EmbeddingService":
    """Return one loaded model for the whole process.

    NodeDatabase and CognitionStore each built their own copy, so a single run
    held two sentence-transformers models in memory at once.
    """
    key = f"{model_name}:{device}"
    if key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[key] = EmbeddingService(model_name=model_name, device=device)
    return _EMBEDDING_CACHE[key]


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

class FAISSIndex:
    """Manage FAISS-based vector indexing for similarity search."""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "IP",
        storage_path: Optional[Path] = None,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None
        self.lock = RLock()

        self.index = faiss.IndexFlatIP(dimension) if index_type == "IP" else faiss.IndexFlatL2(dimension)
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}
        self.next_idx = 0
        # Kept so a removal can rebuild the index: IndexFlat* has no delete.
        self._vectors: Dict[int, np.ndarray] = {}

        if self.storage_path:
            self._load()

    def add(self, node_id: int, vector: np.ndarray) -> None:
        with self.lock:
            if node_id in self.id_to_idx:
                return
            v = self._normalize(vector.reshape(1, -1).astype(np.float32))
            self.index.add(v)
            self._vectors[node_id] = v[0]
            self.id_to_idx[node_id] = self.next_idx
            self.idx_to_id[self.next_idx] = node_id
            self.next_idx += 1

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[int, float]]:
        with self.lock:
            if self.index.ntotal == 0:
                return []
            qv = self._normalize(query_vector.reshape(1, -1).astype(np.float32))
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(qv, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < score_threshold:
                    continue
                nid = self.idx_to_id.get(idx)
                if nid is not None:
                    results.append((nid, float(score)))
            return results

    def remove(self, node_id: int) -> None:
        """Drop a vector for real.

        The previous version only deleted the id mapping, so the vector stayed
        in the index and still consumed one of the ``top_k`` result slots —
        searches silently came back short.
        """
        with self.lock:
            if node_id not in self.id_to_idx:
                return
            self._vectors.pop(node_id, None)
            self._rebuild()

    def _rebuild(self) -> None:
        """Recreate the index from the surviving vectors."""
        self.index = (
            faiss.IndexFlatIP(self.dimension)
            if self.index_type == "IP"
            else faiss.IndexFlatL2(self.dimension)
        )
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        self.next_idx = 0
        for node_id, vector in self._vectors.items():
            self.index.add(vector.reshape(1, -1))
            self.id_to_idx[node_id] = self.next_idx
            self.idx_to_id[self.next_idx] = node_id
            self.next_idx += 1

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if self.index_type == "IP":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return vectors / norms
        return vectors

    def save(self) -> None:
        if not self.storage_path:
            return
        with self.lock:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.storage_path / "faiss.index"))
            meta = {
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "next_idx": self.next_idx,
            }
            # JSON, not pickle: loading a pickle an attacker can write is
            # arbitrary code execution.
            (self.storage_path / "faiss_meta.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )

    def _load(self) -> None:
        if not self.storage_path:
            return
        index_file = self.storage_path / "faiss.index"
        meta_file = self.storage_path / "faiss_meta.json"
        if not index_file.exists() or not meta_file.exists():
            return
        with self.lock:
            self.index = faiss.read_index(str(index_file))
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.id_to_idx = {int(k): v for k, v in meta["id_to_idx"].items()}
            self.idx_to_id = {int(k): v for k, v in meta["idx_to_id"].items()}
            self.next_idx = meta["next_idx"]
            # Rebuild the vector cache so remove() can rebuild the index later.
            self._vectors = {}
            for idx in range(self.index.ntotal):
                node_id = self.idx_to_id.get(idx)
                if node_id is not None:
                    self._vectors[node_id] = self.index.reconstruct(idx)

    def reset(self) -> None:
        with self.lock:
            self.index = faiss.IndexFlatIP(self.dimension) if self.index_type == "IP" else faiss.IndexFlatL2(self.dimension)
            self.id_to_idx.clear()
            self.idx_to_id.clear()
            self._vectors.clear()
            self.next_idx = 0
            if self.storage_path:
                for f in self.storage_path.glob("faiss*"):
                    f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# UCB1 sampler (simplified from ASI-Evolve database/algorithms.py)
# ---------------------------------------------------------------------------

class UCB1Sampler:
    """Upper Confidence Bound 1 sampling for node exploration."""

    def __init__(self, c: float = 1.414):
        self.c = c
        self.total_visits = 0

    def sample(self, nodes: List[Node], n: int) -> List[Node]:
        if not nodes:
            return []
        n = min(n, len(nodes))
        if self.total_visits == 0:
            return nodes[:n]
        scored = []
        for node in nodes:
            exploitation = node.score
            exploration = self.c * math.sqrt(math.log(self.total_visits + 1) / (node.visit_count + 1))
            scored.append((node, exploitation + exploration))
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [node for node, _ in scored[:n]]
        for node in selected:
            node.visit_count += 1
        self.total_visits += 1
        return selected

    def on_node_added(self, node: Node) -> None:
        pass

    def on_node_removed(self, node: Node) -> None:
        pass


# ---------------------------------------------------------------------------
# Node database
# ---------------------------------------------------------------------------

class NodeDatabase:
    """
    Persistent node store.

    Stores nodes as JSON on disk; uses FAISS for similarity search
    and UCB1 for exploration-exploitation sampling.
    """

    def __init__(
        self,
        storage_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_size: Optional[int] = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.nodes: Dict[int, Node] = {}
        self.next_id = 0
        self.max_size = max_size

        # Defer heavy model loading to first use (lazy-init) so FastAPI startup
        # is not blocked by sentence-transformers model downloads (~200 MB).
        self._embedding_model_name = embedding_model
        self._embedding_dim = embedding_dim
        self._faiss: Optional["FAISSIndex"] = None

        self.sampler = UCB1Sampler()
        self._load()

    @property
    def embedding(self) -> "EmbeddingService":
        """Lazy-load the shared EmbeddingService on first use."""
        return get_embedding_service(self._embedding_model_name)

    @property
    def faiss(self) -> Optional["FAISSIndex"]:
        """Lazy-load FAISSIndex on first use. None when faiss is not installed."""
        if self._faiss is None and FAISS_AVAILABLE:
            self._faiss = FAISSIndex(
                dimension=self._embedding_dim,
                storage_path=self.storage_dir / "faiss",
            )
        return self._faiss

    def sample(self, n: int) -> List[Node]:
        with self.lock:
            nodes = list(self.nodes.values())
            selected = self.sampler.sample(nodes, n)
            # Note: visit_count updates are in-memory only; we persist on add(),
            # not on every sample() call, to avoid repeated disk writes.
            return selected

    def add(self, node: Node) -> int:
        with self.lock:
            if self.max_size and len(self.nodes) >= self.max_size:
                self._remove_worst()
            node.id = self.next_id
            self.next_id += 1
            self.nodes[node.id] = node
            self.sampler.on_node_added(node)
            self._index_node(node)
            self._save()
            return node.id

    def _index_node(self, node: Node) -> None:
        """Best-effort vector indexing.

        The embedding stack is optional (see the ImportError guards above), so a
        missing faiss / sentence-transformers must degrade to "no semantic
        search" rather than abort the run.
        """
        text = node.get_context_text()
        if not text:
            return
        try:
            index = self.faiss
            if index is None:
                raise ImportError("faiss-cpu is not installed")
            vector = self.embedding.encode(text)
            index.add(node.id, vector[0])
        except (ImportError, OSError, RuntimeError) as exc:
            _warn_embedding_unavailable(exc)

    def get_all(self) -> List[Node]:
        return list(self.nodes.values())

    def get_best(self) -> Optional[Node]:
        if not self.nodes:
            return None
        return max(self.nodes.values(), key=lambda n: n.score)

    def _remove_worst(self) -> None:
        if not self.nodes:
            return
        worst_id = min(self.nodes, key=lambda nid: (self.nodes[nid].score, nid))
        node = self.nodes.pop(worst_id)
        self.sampler.on_node_removed(node)
        index = self.faiss
        if index is not None:
            index.remove(worst_id)

    def _save(self) -> None:
        data = {
            "next_id": self.next_id,
            "nodes": {str(k): v.to_dict() for k, v in self.nodes.items()},
        }
        _write_json_atomic(self.storage_dir / "nodes.json", data)
        index = self.faiss
        if index is not None:
            index.save()

    def _load(self) -> None:
        data_file = self.storage_dir / "nodes.json"
        if not data_file.exists():
            return
        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)
        self.next_id = data.get("next_id", 0)
        for node_id, node_data in data.get("nodes", {}).items():
            self.nodes[int(node_id)] = Node.from_dict(node_data)

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Cognition store
# ---------------------------------------------------------------------------

class CognitionStore:
    """
    Persistent cognition store with embedding-based retrieval.
    """

    def __init__(
        self,
        storage_dir: Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        retrieval_top_k: int = 5,
        score_threshold: float = 0.5,
        max_size: int = 500,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.retrieval_top_k = retrieval_top_k
        self.score_threshold = score_threshold
        self.max_size = max_size
        self.items: Dict[str, CognitionItem] = {}
        self._content_keys: Dict[str, str] = {}

        # Lazy-init: defer sentence-transformers and FAISS loading to first use
        # (same pattern as NodeDatabase) to avoid blocking FastAPI startup.
        self._embedding_model_name = embedding_model
        self._embedding_dim = embedding_dim
        self._faiss_index: Optional["FAISSIndex"] = None

        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}
        self.next_int_id = 0
        self._load()

    @property
    def embedding(self) -> "EmbeddingService":
        return get_embedding_service(self._embedding_model_name)

    @property
    def faiss(self) -> Optional["FAISSIndex"]:
        if self._faiss_index is None and FAISS_AVAILABLE:
            self._faiss_index = FAISSIndex(
                dimension=self._embedding_dim,
                storage_path=self.storage_dir / "faiss",
            )
        return self._faiss_index

    def add(self, item: CognitionItem) -> str:
        with self.lock:
            if not item.id:
                item.id = str(uuid.uuid4())

            # Every analysis used to be stored verbatim, so after a dozen
            # iterations the store was mostly near-identical lines drowning out
            # whatever was actually new.
            key = " ".join(item.content.split())
            if key in self._content_keys:
                return self._content_keys[key]
            self._content_keys[key] = item.id

            self.items[item.id] = item
            if item.content:
                self._index_item(item)

            while self.max_size and len(self.items) > self.max_size:
                self._evict_oldest()
            self._save()
            return item.id

    def _evict_oldest(self) -> None:
        """Drop the oldest item so the store cannot grow without bound."""
        oldest_id = next(iter(self.items), None)
        if oldest_id is None:
            return
        self.items.pop(oldest_id, None)

        # Look the int id up *before* dropping the mapping: _get_int_id would
        # otherwise mint a fresh one for an item that no longer exists.
        int_id = self.str_to_int.pop(oldest_id, None)
        if int_id is None:
            return
        self.int_to_str.pop(int_id, None)

        index = self.faiss
        if index is not None:
            index.remove(int_id)

    def _index_item(self, item: CognitionItem) -> None:
        try:
            index = self.faiss
            if index is None:
                raise ImportError("faiss-cpu is not installed")
            int_id = self._get_int_id(item.id)
            vector = self.embedding.encode(item.content)
            index.add(int_id, vector[0])
        except (ImportError, OSError, RuntimeError) as exc:
            _warn_embedding_unavailable(exc)

    def search(self, query: str, top_k: Optional[int] = None) -> List[CognitionItem]:
        top_k = top_k or self.retrieval_top_k
        with self.lock:
            try:
                index = self.faiss
                if index is None:
                    raise ImportError("faiss-cpu is not installed")
                qv = self.embedding.encode(query)
                results = index.search(qv[0], top_k, self.score_threshold)
            except (ImportError, OSError, RuntimeError) as exc:
                _warn_embedding_unavailable(exc)
                return []

            items = []
            for int_id, _ in results:
                str_id = self.int_to_str.get(int_id)
                if str_id:
                    item = self.items.get(str_id)
                    if item:
                        items.append(item)
            return items

    def _get_int_id(self, str_id: str) -> int:
        if str_id not in self.str_to_int:
            self.str_to_int[str_id] = self.next_int_id
            self.int_to_str[self.next_int_id] = str_id
            self.next_int_id += 1
        return self.str_to_int[str_id]

    def _save(self) -> None:
        data = {
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "str_to_int": self.str_to_int,
            "int_to_str": {str(k): v for k, v in self.int_to_str.items()},
            "next_int_id": self.next_int_id,
        }
        _write_json_atomic(self.storage_dir / "cognition.json", data)
        index = self.faiss
        if index is not None:
            index.save()

    def _load(self) -> None:
        data_file = self.storage_dir / "cognition.json"
        if not data_file.exists():
            return
        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)
        for item_id, item_data in data.get("items", {}).items():
            self.items[item_id] = CognitionItem.from_dict(item_data)
        self.str_to_int = data.get("str_to_int", {})
        self.int_to_str = {int(k): v for k, v in data.get("int_to_str", {}).items()}
        self.next_int_id = data.get("next_int_id", 0)

        # Rebuild the dedup index so reloads keep rejecting repeats.
        self._content_keys = {
            " ".join(item.content.split()): item_id
            for item_id, item in self.items.items()
            if item.content
        }

    def __len__(self) -> int:
        return len(self.items)
