"""Ground natural-language names to KG entity / relation ids.

Order of preference:
1. exact match after normalization (case, underscores, whitespace) -> score 1.0,
2. bge-m3 nearest neighbours from the precomputed embedding stores.

Results are cached per (kind, normalized text); ``prefetch`` batch-encodes
uncached strings so that RL rollouts pay one encoder call per batch.
"""

import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tkgqa.kg import TemporalKG

KINDS = ("entity", "relation")


@dataclass(frozen=True)
class Candidate:
    id: int
    text: str
    score: float
    exact: bool


def normalize_name(text: str) -> str:
    return " ".join(str(text).replace("_", " ").split()).casefold()


class Grounder:
    def __init__(self, kg: TemporalKG, stores: Optional[Dict[str, object]] = None, encoder=None, top_k: int = 3):
        """``stores``: {"entity": EmbeddingStore, "relation": EmbeddingStore}; omit both for exact-only grounding."""
        self.kg = kg
        self.encoder = encoder
        self.stores = dict(stores or {})
        self.top_k = int(top_k)
        self._texts = {"entity": kg.entity_texts, "relation": kg.relation_texts}
        self._exact: Dict[str, Dict[str, List[int]]] = {}
        for kind in KINDS:
            table: Dict[str, List[int]] = defaultdict(list)
            for i, text in enumerate(self._texts[kind]):
                table[normalize_name(text)].append(i)
            self._exact[kind] = dict(table)
        names = {"entity": kg.entity_names, "relation": kg.relation_names}
        for kind, store in self.stores.items():
            store_names = [item["name"] for item in store.items]
            if store_names != names[kind]:
                raise ValueError(f"{kind} store ids do not match the KG vocabulary; rebuild the store")
        if self.stores and encoder is None:
            raise ValueError("Embedding stores need an encoder")
        self._cache: Dict[Tuple[str, str], List[Candidate]] = {}
        self._embeddings: Dict[Tuple[str, str], object] = {}  # query embeddings (encoder device)
        self._freq: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()  # sessions in worker threads share one encoder and cache

    @classmethod
    def from_store_root(
        cls,
        kg: TemporalKG,
        store_root: str,
        model: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        top_k: int = 3,
        encoder=None,
    ) -> "Grounder":
        from tkgqa.embedding_store import BGEM3Encoder, EmbeddingStore, store_dir

        stores = {kind: EmbeddingStore.load(store_dir(store_root, kg.dataset, kind, model), device=device) for kind in KINDS}
        encoder = encoder or BGEM3Encoder(model, device=device)
        return cls(kg, stores=stores, encoder=encoder, top_k=top_k)

    def _exact_candidates(self, text: str, kind: str) -> List[Candidate]:
        ids = self._exact[kind].get(normalize_name(text), [])
        return [Candidate(i, self._texts[kind][i], 1.0, True) for i in ids]

    def prefetch(self, requests: Iterable[Tuple[str, str]]) -> None:
        """Encode all uncached, non-exact ``(text, kind)`` pairs in one batch per kind."""
        # kind -> {normalized key: text to encode}; the encoder sees the original casing.
        pending: Dict[str, Dict[str, str]] = defaultdict(dict)
        for text, kind in requests:
            key = (kind, normalize_name(text))
            if key in self._cache or key[1] in pending[kind]:
                continue
            exact = self._exact_candidates(text, kind)
            if exact or kind not in self.stores:
                self._cache[key] = exact
            else:
                pending[kind][key[1]] = " ".join(str(text).replace("_", " ").split())
        for kind, table in pending.items():
            if not table:
                continue
            keys = list(table)
            embeddings = self.encoder.encode([table[k] for k in keys])
            for key_text, row in zip(keys, embeddings):
                self._embeddings[(kind, key_text)] = row
            hits = self.stores[kind].search(embeddings, top_k=self.top_k)
            for key_text, row in zip(keys, hits):
                self._cache[(kind, key_text)] = [
                    Candidate(int(h["id"]), self._texts[kind][int(h["id"])], float(h["score"]), False) for h in row
                ]

    def candidates(self, text: str, kind: str) -> List[Candidate]:
        """Exact matches if any, otherwise the top-k embedding neighbours (best first)."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        key = (kind, normalize_name(text))
        with self._lock:
            if key not in self._cache:
                self.prefetch([(text, kind)])
            return self._cache[key]

    def rank(self, text: str, kind: str, ids: Sequence[int], k: int = 5) -> List[int]:
        """Order ``ids`` by similarity to ``text`` (or by KG frequency without an encoder); top ``k``."""
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return []
        if kind in self.stores:
            import torch

            key = (kind, normalize_name(text))
            with self._lock:
                if key not in self._embeddings:
                    self._embeddings[key] = self.encoder.encode([" ".join(str(text).replace("_", " ").split())])[0]
            store = self.stores[kind]
            rows = store.embeddings[torch.as_tensor(ids, device=store.embeddings.device)]
            query = self._embeddings[key].to(device=rows.device, dtype=rows.dtype)
            scores = (rows @ query).float().cpu().numpy()
        else:
            scores = self._frequency(kind)[ids].astype(np.float64)
        order = np.argsort(-scores, kind="stable")[:k]
        return [int(ids[i]) for i in order]

    def _frequency(self, kind: str) -> np.ndarray:
        if kind not in self._freq:
            kg = self.kg
            if kind == "relation":
                self._freq[kind] = np.bincount(kg.rel, minlength=len(kg.relation_texts))
            else:
                n = len(kg.entity_texts)
                self._freq[kind] = np.bincount(kg.subj, minlength=n) + np.bincount(kg.obj, minlength=n)
        return self._freq[kind]

    def cache_size(self) -> int:
        return len(self._cache)
