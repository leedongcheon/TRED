"""Entity / relation vocabularies of the temporal KGs.

Each vocabulary item keeps three things:
- ``name``: the canonical KG token (underscored, as it appears in the KG files),
- ``text``: the natural-language surface form used for embedding and prompts,
- ``freq``: how many KG facts mention the item (useful for tie-breaking later).

Item ids are positions in the returned lists, so they are stable as long as
the source files do not change.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

DATASET_ALIASES = {
    "multitq": "MultiTQ",
    "timelinecronqr": "TimelineCronQR",
    "tcr": "TimelineCronQR",
}
DATASETS = ("MultiTQ", "TimelineCronQR")


@dataclass(frozen=True)
class VocabItem:
    name: str
    text: str
    freq: int


@dataclass(frozen=True)
class KGVocab:
    entity: List[VocabItem]
    relation: List[VocabItem]
    num_facts: int
    dropped_facts: int  # facts skipped because a subject/object name is empty

    def items(self, kind: str) -> List[VocabItem]:
        if kind not in ("entity", "relation"):
            raise ValueError(f"Unknown vocab kind {kind!r}")
        return getattr(self, kind)


def resolve_dataset_name(dataset: str) -> str:
    key = str(dataset).strip()
    resolved = DATASET_ALIASES.get(key.lower(), key)
    if resolved not in DATASETS:
        raise ValueError(f"Unknown TKGQA dataset {dataset!r}; expected one of {DATASETS} (or alias 'tcr').")
    return resolved


def normalize_multitq_token(raw: str) -> str:
    """Undo CSV quoting in MultiTQ KG files.

    full.txt stores five entities CSV-escaped, e.g. ``"Austin_""Jack""_Warner"``,
    while entity2id.json uses ``Austin_"Jack"_Warner``.
    """
    if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].replace('""', '"')
    return raw


def underscore_to_text(name: str) -> str:
    return name.replace("_", " ")


def _load_multitq(dataset_dir: str) -> KGVocab:
    kg_dir = os.path.join(dataset_dir, "kg")
    with open(os.path.join(kg_dir, "entity2id.json"), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(os.path.join(kg_dir, "relation2id.json"), encoding="utf-8") as f:
        relation2id = json.load(f)

    entity_freq: Counter = Counter()
    relation_freq: Counter = Counter()
    num_facts = 0
    with open(os.path.join(kg_dir, "full.txt"), encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            num_facts += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"MultiTQ full.txt:{line_no}: expected 4 columns, got {len(parts)}")
            subj = normalize_multitq_token(parts[0])
            obj = normalize_multitq_token(parts[2])
            entity_freq[subj] += 1
            entity_freq[obj] += 1
            relation_freq[parts[1]] += 1

    unknown_entities = set(entity_freq) - set(entity2id)
    unknown_relations = set(relation_freq) - set(relation2id)
    if unknown_entities or unknown_relations:
        raise ValueError(
            f"MultiTQ KG has tokens missing from the id maps: "
            f"entities={sorted(unknown_entities)[:5]}, relations={sorted(unknown_relations)[:5]}"
        )

    def _ordered(name_to_id: Dict[str, int], freq: Counter) -> List[VocabItem]:
        names = sorted(name_to_id, key=name_to_id.__getitem__)
        if [name_to_id[n] for n in names] != list(range(len(names))):
            raise ValueError("MultiTQ id map is not contiguous from 0.")
        return [VocabItem(name=n, text=underscore_to_text(n), freq=int(freq.get(n, 0))) for n in names]

    return KGVocab(
        entity=_ordered(entity2id, entity_freq),
        relation=_ordered(relation2id, relation_freq),
        num_facts=num_facts,
        dropped_facts=0,
    )


def _load_timelinecronqr(dataset_dir: str) -> KGVocab:
    kg_dir = os.path.join(dataset_dir, "kg")
    entity_text: Dict[str, str] = {}
    relation_text: Dict[str, str] = {}
    entity_freq: Counter = Counter()
    relation_freq: Counter = Counter()
    num_facts = 0
    dropped_facts = 0

    def _register(table: Dict[str, str], name: str, text: str, where: str) -> None:
        previous = table.setdefault(name, text)
        if previous != text:
            raise ValueError(f"{where}: {name!r} maps to both {previous!r} and {text!r}")

    with open(os.path.join(kg_dir, "full.txt"), encoding="utf-8") as f_kg, open(
        os.path.join(kg_dir, "full_no_underscore.txt"), encoding="utf-8"
    ) as f_txt:
        for line_no, (line_kg, line_txt) in enumerate(zip(f_kg, f_txt), 1):
            kg = line_kg.rstrip("\n").split("\t")
            txt = line_txt.rstrip("\n").split("\t")
            if len(kg) != 5 or len(txt) != 5:
                raise ValueError(f"TimelineCronQR full.txt:{line_no}: expected 5 columns")
            num_facts += 1
            # 27 position_held facts have an empty object label; no question uses them.
            if not kg[0] or not kg[2]:
                dropped_facts += 1
                continue
            where = f"TimelineCronQR full.txt:{line_no}"
            _register(entity_text, kg[0], txt[0], where)
            _register(entity_text, kg[2], txt[2], where)
            _register(relation_text, kg[1], txt[1], where)
            entity_freq[kg[0]] += 1
            entity_freq[kg[2]] += 1
            relation_freq[kg[1]] += 1

    def _ordered(table: Dict[str, str], freq: Counter) -> List[VocabItem]:
        return [VocabItem(name=n, text=table[n], freq=int(freq[n])) for n in sorted(table)]

    return KGVocab(
        entity=_ordered(entity_text, entity_freq),
        relation=_ordered(relation_text, relation_freq),
        num_facts=num_facts,
        dropped_facts=dropped_facts,
    )


_LOADERS = {
    "MultiTQ": _load_multitq,
    "TimelineCronQR": _load_timelinecronqr,
}


def load_vocab(dataset: str, data_root: str) -> KGVocab:
    """Return the entity / relation vocabularies of one dataset."""
    dataset = resolve_dataset_name(dataset)
    vocab = _LOADERS[dataset](os.path.join(data_root, dataset))
    for kind in ("entity", "relation"):
        items = vocab.items(kind)
        if any(not item.text.strip() for item in items):
            raise ValueError(f"{dataset} {kind} vocabulary contains an empty text")
        texts = Counter(item.text for item in items)
        duplicated = [t for t, c in texts.items() if c > 1]
        if duplicated:
            raise ValueError(f"{dataset} {kind} texts are not unique, e.g. {duplicated[:5]}")
    return vocab
