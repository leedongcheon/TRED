"""bge-m3 dense embedding stores for entity / relation grounding.

Store layout (one directory per dataset x kind)::

    {store_root}/{dataset}/{model_tag}/{kind}/
        embeddings.npy   # (N, D) L2-normalized, row i <-> items.jsonl line i
        items.jsonl      # {"id", "name", "text", "freq"} per line
        manifest.json    # model, pooling, dim, count, source files, ...
"""

import json
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

DEFAULT_MODEL = "BAAI/bge-m3"
EMBEDDINGS_FILE = "embeddings.npy"
ITEMS_FILE = "items.jsonl"
MANIFEST_FILE = "manifest.json"
FORMAT_VERSION = 1
KINDS = ("entity", "relation")


def model_tag(model_name_or_path: str) -> str:
    return os.path.basename(str(model_name_or_path).rstrip("/"))


def store_dir(store_root: str, dataset: str, kind: str, model: str = DEFAULT_MODEL) -> str:
    if kind not in KINDS:
        raise ValueError(f"Unknown store kind {kind!r}; expected one of {KINDS}.")
    return os.path.join(store_root, dataset, model_tag(model), kind)


class BGEM3Encoder:
    """Dense bge-m3 encoder.

    Matches the dense output of ``BGEM3FlagModel``: CLS token of the last
    hidden state, L2-normalized. bge-m3 needs no query instruction prefix, so
    queries and KG items share this encoder.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        max_length: int = 128,
        local_files_only: bool = True,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            dtype = torch.float32
        self.model_name_or_path = model_name_or_path
        self.max_length = int(max_length)
        # Load from a local directory: with a hub id, transformers 4.57 tokenizers
        # query the hub (model_info) even when the files are cached.
        if os.path.isdir(model_name_or_path):
            self.local_path = model_name_or_path
        else:
            from huggingface_hub import snapshot_download

            self.local_path = snapshot_download(model_name_or_path, local_files_only=local_files_only)
        self.revision = os.path.basename(os.path.normpath(self.local_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_path)
        self.model = AutoModel.from_pretrained(self.local_path, dtype=dtype).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def count_truncated(self, texts: Sequence[str]) -> int:
        lengths = self.tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
        return int(sum(len(ids) > self.max_length for ids in lengths))

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 256, show_progress: bool = False) -> torch.Tensor:
        """Return a float32 (N, D) tensor of L2-normalized embeddings on ``self.device``."""
        texts = [str(t) for t in texts]
        out = torch.empty((len(texts), self.dim), dtype=torch.float32, device=self.device)
        if not texts:
            return out
        # Longest first so each batch pads to a similar length.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        starts = range(0, len(order), batch_size)
        if show_progress:
            from tqdm import tqdm

            starts = tqdm(starts, total=(len(order) + batch_size - 1) // batch_size, desc="encode")
        for start in starts:
            idx = order[start : start + batch_size]
            batch = self.tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            cls = self.model(**batch).last_hidden_state[:, 0].float()
            out[torch.as_tensor(idx, device=self.device)] = torch.nn.functional.normalize(cls, dim=-1)
        return out


class EmbeddingStore:
    """Read-only nearest-neighbour store over L2-normalized embeddings (cosine = dot product)."""

    def __init__(self, items: List[Dict[str, object]], embeddings: torch.Tensor, manifest: Dict[str, object]):
        if embeddings.dim() != 2 or embeddings.shape[0] != len(items):
            raise ValueError(f"embeddings shape {tuple(embeddings.shape)} does not match {len(items)} items")
        self.items = items
        self.embeddings = embeddings
        self.manifest = manifest
        self._name_to_id = {str(item["name"]): i for i, item in enumerate(items)}

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "EmbeddingStore":
        with open(os.path.join(path, MANIFEST_FILE), encoding="utf-8") as f:
            manifest = json.load(f)
        with open(os.path.join(path, ITEMS_FILE), encoding="utf-8") as f:
            items = [json.loads(line) for line in f]
        if any(int(item["id"]) != i for i, item in enumerate(items)):
            raise ValueError(f"{path}/{ITEMS_FILE}: ids are not 0..N-1 in order")
        array = np.load(os.path.join(path, EMBEDDINGS_FILE))
        if int(manifest["count"]) != len(items) or array.shape != (len(items), int(manifest["dim"])):
            raise ValueError(f"{path}: manifest/items/embeddings sizes disagree")
        torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # fp16 matmul is fine on GPU; keep float32 on CPU.
        dtype = torch.float16 if torch_device.type == "cuda" else torch.float32
        embeddings = torch.from_numpy(array).to(device=torch_device, dtype=dtype)
        return cls(items=items, embeddings=embeddings, manifest=manifest)

    def id_of(self, name: str) -> Optional[int]:
        return self._name_to_id.get(name)

    def search(self, query_embeddings: torch.Tensor, top_k: int = 10) -> List[List[Dict[str, object]]]:
        """Top-k items per query; each hit is ``{"id", "name", "text", "freq", "score"}``."""
        queries = query_embeddings
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        queries = queries.to(device=self.embeddings.device, dtype=self.embeddings.dtype)
        k = min(int(top_k), len(self.items))
        scores, indices = (queries @ self.embeddings.T).float().topk(k, dim=-1)
        results = []
        for row_scores, row_indices in zip(scores.tolist(), indices.tolist()):
            results.append([{**self.items[i], "score": float(s)} for s, i in zip(row_scores, row_indices)])
        return results

    def search_texts(
        self,
        encoder: BGEM3Encoder,
        queries: Sequence[str],
        top_k: int = 10,
        batch_size: int = 256,
    ) -> List[List[Dict[str, object]]]:
        if model_tag(encoder.model_name_or_path) != model_tag(str(self.manifest["model"])):
            raise ValueError(
                f"Encoder {encoder.model_name_or_path!r} does not match store model {self.manifest['model']!r}"
            )
        return self.search(encoder.encode(queries, batch_size=batch_size), top_k=top_k)


def write_store(
    path: str,
    items: List[Dict[str, object]],
    embeddings: np.ndarray,
    manifest: Dict[str, object],
    overwrite: bool = False,
) -> None:
    """Write a store atomically: build in ``path.tmp`` then rename."""
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"Store already exists: {path} (pass overwrite=True to replace it)")
    tmp_path = f"{path}.tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)
    np.save(os.path.join(tmp_path, EMBEDDINGS_FILE), embeddings)
    with open(os.path.join(tmp_path, ITEMS_FILE), "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(os.path.join(tmp_path, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump({"format_version": FORMAT_VERSION, **manifest}, f, ensure_ascii=False, indent=2)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp_path, path)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
