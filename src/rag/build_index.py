import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Optional

from .faiss_index import FAISSIndex
from .user_store import UserStore

_model: SentenceTransformer | None = None

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 100


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_embedding(text: str | list[str]) -> list[float] | list[list[float]]:
    model = _get_model()
    result = model.encode(text, show_progress_bar=False)
    if isinstance(text, list):
        return result.tolist()
    return result.tolist()


def build_user_store(
    data: pd.DataFrame,
    output_dir: str,
) -> tuple[UserStore, list[list[float]]]:
    user_store_path = os.path.join(output_dir, "user_store.jsonl")
    user_store = UserStore(user_store_path)

    posts_list = data["text"].astype(str).tolist()
    user_ids = [f"user_{i}" for i in range(len(data))]

    posts_embedding_list = []
    for i in range(0, len(posts_list), BATCH_SIZE):
        batch = posts_list[i: i + BATCH_SIZE]
        batch_emb = get_embedding(batch)
        if isinstance(batch_emb, list) and isinstance(batch_emb[0], list):
            posts_embedding_list.extend(batch_emb)
        else:
            posts_embedding_list.append(batch_emb)

    for uid, posts, emb in zip(user_ids, posts_list, posts_embedding_list):
        user_store.add(uid, posts, emb)
    user_store.save()

    return user_store, posts_embedding_list


def build_trait_index(
    posts_embeddings: list[list[float]],
    data: pd.DataFrame,
    trait: str,
    label_col: str,
    output_dir: str,
) -> None:
    meta_path = os.path.join(output_dir, f"{trait}_meta.jsonl")
    index_path = os.path.join(output_dir, f"{trait}.faiss")
    if os.path.exists(meta_path) and os.path.exists(index_path):
        return

    records = []
    for i, (_, row) in enumerate(data.iterrows()):
        records.append({
            "user_id": f"user_{i}",
            "posts_raw": str(row["text"]),
            "label": str(row[label_col]),
        })

    texts = [r["posts_raw"] for r in records]
    text_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        batch_emb = get_embedding(batch)
        if isinstance(batch_emb, list) and isinstance(batch_emb[0], list):
            text_embeddings.extend(batch_emb)
        else:
            text_embeddings.append(batch_emb)

    combined_vectors = []
    for posts_emb, text_emb in zip(posts_embeddings, text_embeddings):
        if isinstance(posts_emb, np.ndarray):
            posts_emb = posts_emb.tolist()
        if isinstance(text_emb, np.ndarray):
            text_emb = text_emb.tolist()
        avg = [(p + t) / 2 for p, t in zip(posts_emb, text_emb)]
        combined_vectors.append(avg)

    vectors = np.array(combined_vectors, dtype="float32")
    faiss_index = FAISSIndex(dimension=vectors.shape[1])
    faiss_index.build(vectors, records)
    faiss_index.save(index_path, meta_path)


def build_vector_db(
    data: pd.DataFrame,
    output_dir: str = "data/vector_db/essays",
    trait: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    user_store, posts_embedding_list = build_user_store(data, output_dir)

    TRAIT_LABEL_COLS = {
        "Extraversion": "cEXT",
        "Neuroticism": "cNEU",
        "Agreeableness": "cAGR",
        "Conscientiousness": "cCON",
        "Openness to Experience": "cOPN",
    }

    traits_to_process = [trait] if trait else list(TRAIT_LABEL_COLS.keys())
    for t in traits_to_process:
        label_col = TRAIT_LABEL_COLS[t]
        build_trait_index(posts_embedding_list, data, t, label_col, output_dir)

    return user_store, posts_embedding_list
