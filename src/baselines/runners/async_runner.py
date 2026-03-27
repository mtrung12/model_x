import asyncio
import os
import pandas as pd

from src.clients.gpt_client import gpt_call_async
from src.clients.llama_client import llama_call_async
from src.common.parsers import parse_llm_output
from src.common.prompts.baseline import (
    NORMAL_USER_PROMPT,
    COT_SYS_PROMPT,
    ONESHOT_SYS_PROMPT,
    ZEROSHOT_SYS_PROMPT,
)
from src.common.trait_defs import TRAIT_COLS, TRAIT_MAP


async def call_async(user_prompt: str, system_prompt: str, model_name: str, max_new_tokens: int, log_filepath: str, record_idx: int = None, trait_col: str = None):
    if model_name.startswith("gpt"):
        return await gpt_call_async(
            user_prompt,
            system_prompt,
            model=model_name,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            top_p=1.0,
            log_filepath=log_filepath,
            record_idx=record_idx,
            trait_col=trait_col,
        )
    return await llama_call_async(user_prompt, system_prompt, model_name, max_new_tokens, log_filepath, record_idx=record_idx, trait_col=trait_col)


async def predict_one_trait(
    text: str,
    trait_col: str,
    model_name: str,
    max_new_tokens: int,
    log_filepath: str,
    prompt_mode: str = "zeroshot",
    example_text: str = None,
    example_label: str = None,
    record_idx: int = None,
):
    trait_name = TRAIT_MAP.get(trait_col, trait_col)

    if prompt_mode == "zeroshot":
        system_prompt = ZEROSHOT_SYS_PROMPT.format(trait=trait_name)
    elif prompt_mode == "oneshot":
        system_prompt = ONESHOT_SYS_PROMPT.format(
            trait=trait_name,
            example_text=example_text,
            example_label=example_label,
        )
    elif prompt_mode == "cot":
        system_prompt = COT_SYS_PROMPT.format(trait=trait_name)
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    user_prompt = NORMAL_USER_PROMPT.format(text=text)
    raw_output = await call_async(user_prompt, system_prompt, model_name, max_new_tokens, log_filepath, record_idx=record_idx, trait_col=trait_col)
    return parse_llm_output(raw_output)


async def run_one(
    idx: int,
    text: str,
    model_name: str,
    max_new_tokens: int,
    log_filepath: str,
    prompt_mode: str = "zeroshot",
    top_k: int = 5,
    vector_db_dir: str = "data/vector_db",
    train_csv: str = None,
):
    results = {}
    errors = []

    example_text = None
    example_label = None
    if prompt_mode == "oneshot" and not train_csv:
        raise ValueError("oneshot mode requires train_csv to be provided")
    if train_csv:
        train_df = pd.read_csv(train_csv)
        example_df = train_df.sample(n=1, random_state=12)
        example_text = example_df.iloc[0]["text"]

    for trait_col in TRAIT_COLS:
        try:
            example_label = example_df.iloc[0][trait_col] if train_csv else None
            result = await predict_one_trait(
                text,
                trait_col,
                model_name,
                max_new_tokens,
                log_filepath,
                prompt_mode,
                example_text,
                example_label,
                record_idx=idx,
            )
            results[trait_col] = result
        except Exception as e:
            errors.append(f"{trait_col}: {e}")
            results[trait_col] = None

    error = "; ".join(errors) if errors else None
    return (idx, results, None, error)


async def run_batch(
    df: pd.DataFrame,
    model_name: str,
    max_new_tokens: int,
    log_filepath: str,
    prompt_mode: str = "zeroshot",
    top_k: int = 5,
    vector_db_dir: str = "data/vector_db",
    max_concurrency: int = 10,
    train_csv: str = None,
):
    n_records = len(df)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(idx: int, row):
        async with semaphore:
            return await run_one(
                idx,
                row["text"],
                model_name,
                max_new_tokens,
                log_filepath,
                prompt_mode,
                top_k=top_k,
                vector_db_dir=vector_db_dir,
                train_csv=train_csv,
            )

    tasks = [bounded(i, df.iloc[i]) for i in range(n_records)]
    completed = await asyncio.gather(*tasks)

    results_list = [None] * n_records
    fail_count = 0

    for idx, results, _, error in completed:
        results_list[idx] = results
        if error:
            fail_count += 1
            print(f"Got error: {error} at record {idx}")

    return results_list, fail_count
