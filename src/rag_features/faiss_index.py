import faiss
import numpy as np
import json
import os
from typing import List, Optional


class FAISSIndex:
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.meta: List[dict] = []
        self.index_path = index_path

    def build(self, vectors: np.ndarray, meta: List[dict]):
        vectors = vectors.astype("float32")
        if vectors.shape[0] == 0:
            raise ValueError("No vectors provided to build index")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match expected {self.dimension}"
            )

        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.meta = meta

    def save(self, index_path: str, meta_path: str):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for item in self.meta:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        self.meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, List[dict]]:
        query_vector = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.meta):
                results.append(
                    {"distance": float(dist), **self.meta[idx]}
                )
        return distances, results
