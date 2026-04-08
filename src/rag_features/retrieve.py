import os
import json
import numpy as np
from typing import Optional

from .faiss_index import FAISSIndex
from .user_store import UserStore
from .build_index import get_embedding, EMBEDDING_MODEL


class RAGRetriever:
    def __init__(self, db_dir: str = "data/vector_db_with_features/essays"):
        self.db_dir = db_dir
        self.user_store = UserStore(
            os.path.join(db_dir, "user_store.jsonl")
        )
        self.user_store.load()
        self.trait_indexes: dict[str, FAISSIndex] = {}

    def load_trait_index(self, trait: str):
        index_path = os.path.join(self.db_dir, f"{trait}.faiss")
        meta_path = os.path.join(self.db_dir, f"{trait}_meta.jsonl")
        idx = FAISSIndex(dimension=0)
        idx.load(index_path, meta_path)
        self.trait_indexes[trait] = idx
        return idx

    def get_trait_index(self, trait: str) -> FAISSIndex:
        if trait not in self.trait_indexes:
            self.load_trait_index(trait)
        return self.trait_indexes[trait]

    def retrieve(
        self,
        posts: str,
        trait: str,
        top_k: int = 5,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> list[dict]:
        query_emb = get_embedding(posts)
        query_emb = np.array(query_emb, dtype="float32")

        index = self.get_trait_index(trait)
        _, results = index.search(query_emb, top_k)

        enriched = []
        for r in results:
            user_id = r["user_id"]
            user_data = self.user_store.get(user_id)
            enriched.append(
                {
                    "user_id": user_id,
                    "posts_raw": user_data["posts_raw"] if user_data else "",
                    "llm_features": r.get("llm_features", ""),
                    "label": r.get("label", ""),
                    "distance": r.get("distance", 0.0),
                }
            )
        return enriched

    def build_context(
        self,
        posts: str,
        trait: str,
        top_k: int = 5,
    ) -> str:
        retrieved = self.retrieve(posts, trait, top_k)

        context_lines = []
        for i, r in enumerate(retrieved, 1):
            context_lines.append(
                f"[User {i}] Posts: \"{r['posts_raw'][:500]}...\" "
                f"Features: \"{r['llm_features']}\" "
                f"Label: {r['label']}"
            )

        context = "\n".join(context_lines)
        context += f"\n\nTarget user posts: \"{posts}\""
        return context
