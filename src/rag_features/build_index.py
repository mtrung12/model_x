import asyncio
import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio
from typing import Optional

from .faiss_index import FAISSIndex
from .user_store import UserStore
from .prompts import TRAITS, build_extraction_messages
from ..common.trait_defs import TRAIT_MAP

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _model

TRAIT_KEY_TO_COL = {
    "Extraversion": "cEXT",
    "Neuroticism": "cNEU",
    "Agreeableness": "cAGR",
    "Conscientiousness": "cCON",
    "Openness to Experience": "cOPN",
}
from ..clients.gpt_client import get_async_client, gpt_call_async


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EXTRACTION_MODEL = "gpt-4o-mini"
EXTRACTION_TEMPERATURE = 0.7
EXTRACTION_MAX_TOKENS = 512
EXTRACTION_TOP_P = 0.9
BATCH_SIZE = 100


def get_embedding(text: str | list[str], client=None) -> list[float] | list[list[float]]:
    model = _get_model()
    result = model.encode(text, show_progress_bar=False)
    if isinstance(text, list):
        return result.tolist()
    return result.tolist()


async def extract_features_batch(
    df: pd.DataFrame,
    trait: str,
    log_dir: Optional[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, dict]:
    client = get_async_client()
    tasks = []
    indices = []

    for idx, row in df.iterrows():
        posts = str(row["text"])
        messages = build_extraction_messages(trait, posts)
        sys_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        log_path = os.path.join(log_dir, f"extract_{trait}.log") if log_dir else ""

        tasks.append(
            gpt_call_async(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                model=EXTRACTION_MODEL,
                temperature=EXTRACTION_TEMPERATURE,
                max_new_tokens=EXTRACTION_MAX_TOKENS,
                top_p=EXTRACTION_TOP_P,
                log_filepath=log_path,
                record_idx=int(idx),
                trait_col=trait,
            )
        )
        indices.append(str(idx))

    results = await tqdm_asyncio.gather(*tasks)
    out = {}
    label_col = TRAIT_KEY_TO_COL.get(trait, trait.replace(" ", "").lower())
    for idx, content in zip(indices, results):
        out[idx] = {"user_id": str(idx), "llm_features": content}
        if label_col in df.columns:
            out[idx]["label"] = str(df.loc[int(idx), label_col])
    return out


async def extract_trait(df: pd.DataFrame, trait: str, log_dir: Optional[str], max_concurrent: int = 10) -> dict[str, dict]:
    semaphore = asyncio.Semaphore(max_concurrent)
    return await extract_features_batch(df, trait, log_dir, semaphore)


def extract_features(
    data: pd.DataFrame,
    trait: str,
    output_dir,
    log_dir,
    max_concurrent: int,
) -> dict[str, dict]:
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    return asyncio.run(extract_trait(data, trait, log_dir or "", max_concurrent))


def save_features(features: dict[str, dict], trait: str, output_dir: str) -> None:
    path = os.path.join(output_dir, f"{trait}_features.jsonl")
    with open(path, "w") as f:
        for record in features.values():
            f.write(json.dumps(record) + "\n")


def load_features(trait: str, output_dir: str) -> dict[str, dict]:
    path = os.path.join(output_dir, f"{trait}_features.jsonl")
    if not os.path.exists(path):
        return None
    features = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            features[rec["user_id"]] = rec
    return features


def build_trait_index(
    posts_embeddings: list[list[float]],
    features: dict[str, dict],
    trait: str,
    output_dir: str,
) -> None:
    meta_path = os.path.join(output_dir, f"{trait}_meta.jsonl")
    if os.path.exists(meta_path):
        return

    meta_records = list(features.values())
    if not meta_records:
        return

    feature_texts = [r["llm_features"] for r in meta_records]
    feature_embeddings = []
    for i in range(0, len(feature_texts), BATCH_SIZE):
        batch = feature_texts[i : i + BATCH_SIZE]
        batch_emb = get_embedding(batch)
        if isinstance(batch_emb, list) and isinstance(batch_emb[0], list):
            feature_embeddings.extend(batch_emb)
        else:
            feature_embeddings.append(batch_emb)

    combined_vectors = []
    for posts_emb, feat_emb in zip(posts_embeddings, feature_embeddings):
        if isinstance(posts_emb, np.ndarray):
            posts_emb = posts_emb.tolist()
        if isinstance(feat_emb, np.ndarray):
            feat_emb = feat_emb.tolist()
        avg = [(p + f) / 2 for p, f in zip(posts_emb, feat_emb)]
        combined_vectors.append(avg)

    vectors = np.array(combined_vectors, dtype="float32")

    enriched_records = []
    for rec, feat_emb in zip(meta_records, feature_embeddings):
        if isinstance(feat_emb, np.ndarray):
            feat_emb = feat_emb.tolist()
        rec["llm_features_embedding"] = feat_emb
        enriched_records.append(rec)

    faiss_index = FAISSIndex(dimension=vectors.shape[1])
    faiss_index.build(vectors, enriched_records)

    index_path = os.path.join(output_dir, f"{trait}.faiss")
    faiss_index.save(index_path, meta_path)


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
        batch = posts_list[i : i + BATCH_SIZE]
        batch_emb = get_embedding(batch)
        if isinstance(batch_emb, list) and isinstance(batch_emb[0], list):
            posts_embedding_list.extend(batch_emb)
        else:
            posts_embedding_list.append(batch_emb)

    for uid, posts, emb in zip(user_ids, posts_list, posts_embedding_list):
        user_store.add(uid, posts, emb)
    user_store.save()

    return user_store, posts_embedding_list


def build_vector_db(
    data: pd.DataFrame,
    output_dir: str = "data/vector_db_with_features/essays",
    embedding_model: str = EMBEDDING_MODEL,
    k: int = 5,
    log_dir: Optional[str] = None,
    max_concurrent: int = 10,
    trait: Optional[str] = None,
):
    """Build vector DB. If trait is specified, only process that trait; otherwise process all."""
    os.makedirs(output_dir, exist_ok=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    user_store, posts_embedding_list = build_user_store(data, output_dir)

    traits_to_process = [trait] if trait else list(TRAITS.keys())

    for t in traits_to_process:
        features = asyncio.run(extract_trait(data, t, log_dir, max_concurrent))
        build_trait_index(posts_embedding_list, features, t, output_dir)

    return user_store, None
