"""Temporal KG store with inverted indexes.

Facts are ``(subj, rel, obj, start, end)`` with day-ordinal dates. Fact ids are
assigned in ``(start, end, subj, rel, obj)`` order, so any sorted id array is
also in chronological order. Entity / relation ids are the ``tkgqa.kg_vocab``
ids, i.e. the row ids of the embedding stores.
"""

import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tkgqa.kg_vocab import load_vocab, normalize_multitq_token, resolve_dataset_name
from tkgqa.timeutil import format_span, gran_key, parse_kg_date

@dataclass(frozen=True)
class Fact:
    id: int
    subj: str
    rel: str
    obj: str
    start: int
    end: int


class _Index:
    """Sorted-key index: ``get(key)`` returns the ascending fact ids with that key."""

    def __init__(self, keys: np.ndarray):
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.ids = order.astype(np.int64)

    def get(self, key: int) -> np.ndarray:
        lo = np.searchsorted(self.keys, key, side="left")
        hi = np.searchsorted(self.keys, key, side="right")
        return self.ids[lo:hi]


class TemporalKG:
    def __init__(
        self,
        dataset: str,
        entity_names: Sequence[str],
        entity_texts: Sequence[str],
        relation_names: Sequence[str],
        relation_texts: Sequence[str],
        rows: np.ndarray,
        stats: Dict[str, int],
        date_precision: str = "day",
    ):
        """``rows``: int64 array of shape (N, 5) with columns subj, rel, obj, start, end."""
        self.dataset = dataset
        self.entity_names = list(entity_names)
        self.entity_texts = list(entity_texts)
        self.relation_names = list(relation_names)
        self.relation_texts = list(relation_texts)
        self.date_precision = date_precision

        # Deduplicate and order facts by (start, end, subj, rel, obj).
        reordered = rows[:, [3, 4, 0, 1, 2]] if len(rows) else rows.reshape(0, 5)
        unique = np.unique(reordered, axis=0)
        self.stats = dict(stats)
        self.stats["duplicates_removed"] = int(len(rows) - len(unique))
        self.stats["facts"] = int(len(unique))
        self.start = unique[:, 0].astype(np.int64)
        self.end = unique[:, 1].astype(np.int64)
        self.subj = unique[:, 2].astype(np.int64)
        self.rel = unique[:, 3].astype(np.int64)
        self.obj = unique[:, 4].astype(np.int64)

        self._gran_keys: Dict[Tuple[str, str], np.ndarray] = {}
        for gran in ("day", "month", "year"):
            for side, values in (("start", self.start), ("end", self.end)):
                uniq, inverse = np.unique(values, return_inverse=True)
                keys = np.asarray([gran_key(int(v), gran) for v in uniq], dtype=np.int64)
                self._gran_keys[(side, gran)] = keys[inverse]

        num_e = max(len(self.entity_names), 1)
        num_r = max(len(self.relation_names), 1)
        self._by_subj = _Index(self.subj)
        self._by_rel = _Index(self.rel)
        self._by_obj = _Index(self.obj)
        self._by_sr = _Index(self.subj * num_r + self.rel)
        self._by_ro = _Index(self.rel * num_e + self.obj)
        self._by_so = _Index(self.subj * num_e + self.obj)
        self._num_e, self._num_r = num_e, num_r

    def __len__(self) -> int:
        return len(self.start)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, dataset: str, data_root: str) -> "TemporalKG":
        dataset = resolve_dataset_name(dataset)
        vocab = load_vocab(dataset, data_root)
        entity_id = {item.name: i for i, item in enumerate(vocab.entity)}
        relation_id = {item.name: i for i, item in enumerate(vocab.relation)}
        stats: Counter = Counter()
        date_cache: Dict[str, int] = {}

        def _date(text: str) -> int:
            value = date_cache.get(text)
            if value is None:
                value = date_cache[text] = parse_kg_date(text)
            return value

        rows: List[Tuple[int, int, int, int, int]] = []
        with open(os.path.join(data_root, dataset, "kg", "full.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                stats["raw_lines"] += 1
                if dataset == "MultiTQ":
                    subj, rel, obj = (normalize_multitq_token(parts[0]), parts[1], normalize_multitq_token(parts[2]))
                    start_text = end_text = parts[3]
                else:
                    subj, rel, obj, start_text, end_text = parts
                    if not subj or not obj:
                        stats["dropped_empty_name"] += 1
                        continue
                try:
                    start, end = _date(start_text), _date(end_text)
                except ValueError:
                    stats["dropped_bad_date"] += 1
                    continue
                if start > end:
                    start, end = end, start
                    stats["swapped_start_end"] += 1
                rows.append((entity_id[subj], relation_id[rel], entity_id[obj], start, end))

        return cls(
            dataset=dataset,
            entity_names=[item.name for item in vocab.entity],
            entity_texts=[item.text for item in vocab.entity],
            relation_names=[item.name for item in vocab.relation],
            relation_texts=[item.text for item in vocab.relation],
            rows=np.asarray(rows, dtype=np.int64).reshape(-1, 5),
            stats=dict(stats),
        )

    @classmethod
    def from_facts(
        cls,
        facts: Sequence[Tuple[str, str, str, str, str]],
        dataset: str = "custom",
        date_precision: str = "day",
    ) -> "TemporalKG":
        """Build a small KG from ``(subj, rel, obj, start, end)`` text tuples (tests, toy examples)."""
        entities = sorted({f[0] for f in facts} | {f[2] for f in facts})
        relations = sorted({f[1] for f in facts})
        e_id = {name: i for i, name in enumerate(entities)}
        r_id = {name: i for i, name in enumerate(relations)}
        rows = []
        for subj, rel, obj, start, end in facts:
            s, e = parse_kg_date(start), parse_kg_date(end)
            rows.append((e_id[subj], r_id[rel], e_id[obj], min(s, e), max(s, e)))
        return cls(
            dataset=dataset,
            entity_names=entities,
            entity_texts=[name.replace("_", " ") for name in entities],
            relation_names=relations,
            relation_texts=[name.replace("_", " ") for name in relations],
            rows=np.asarray(rows, dtype=np.int64).reshape(-1, 5),
            stats={"raw_lines": len(facts)},
            date_precision=date_precision,
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def find(self, subj: Optional[int] = None, rel: Optional[int] = None, obj: Optional[int] = None) -> np.ndarray:
        """Ascending (= chronological) ids of facts matching the given slots; None is a wildcard."""
        if subj is not None and rel is not None:
            ids = self._by_sr.get(subj * self._num_r + rel)
            return ids if obj is None else ids[self.obj[ids] == obj]
        if rel is not None and obj is not None:
            return self._by_ro.get(rel * self._num_e + obj)
        if subj is not None and obj is not None:
            return self._by_so.get(subj * self._num_e + obj)
        if subj is not None:
            return self._by_subj.get(subj)
        if rel is not None:
            return self._by_rel.get(rel)
        if obj is not None:
            return self._by_obj.get(obj)
        raise ValueError("find() needs at least one bound slot")

    def gran_keys(self, side: str, gran: str) -> np.ndarray:
        """Per-fact ``start``/``end`` keys at ``gran`` (see ``timeutil.gran_key``)."""
        return self._gran_keys[(side, gran)]

    def fact(self, fact_id: int) -> Fact:
        i = int(fact_id)
        return Fact(
            id=i,
            subj=self.entity_texts[self.subj[i]],
            rel=self.relation_texts[self.rel[i]],
            obj=self.entity_texts[self.obj[i]],
            start=int(self.start[i]),
            end=int(self.end[i]),
        )

    def format_fact(self, fact_id: int) -> str:
        fact = self.fact(fact_id)
        when = format_span((fact.start, fact.end), self.date_precision, collapse=True)
        return f"({fact.subj}, {fact.rel}, {fact.obj}, {when})"
