import os
import asyncio
import pandas as pd
from typing import Optional

from ...common.trait_defs import TRAIT_COLS, TRAIT_MAP
from ...clients.gpt_client import gpt_call_async
from ...clients.llama_client import llama_call_async
from ..prompts import (
    TRAITS,
    SYS_PROMPT_EXPLAINER,
    USR_PROMPT_EXPLAINER,
    SYS_PROMPT_JUDGE,
    USR_PROMPT_JUDGE,
)
from ..retriever import RAGRetriever


EXPLAINER_TEMPERATURE = 0.7
EXPLAINER_MAX_TOKENS = 512
EXPLAINER_TOP_P = 0.9
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 256
JUDGE_TOP_P = 1.0


async def _call_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    log_filepath: Optional[str],
    record_idx: Optional[int],
    trait_col: Optional[str],
):
    if model_name.startswith("gpt"):
        return await gpt_call_async(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            log_filepath=log_filepath,
            record_idx=record_idx,
            trait_col=trait_col,
        )
    return await llama_call_async(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        log_filepath=log_filepath,
        record_idx=record_idx,
        trait_col=trait_col,
    )


def _build_explainer_prompts(trait: str, trait_level: str, user_text: str, sim_text: str):
    trait_desc = TRAITS[trait][trait_level]
    sys_prompt = SYS_PROMPT_EXPLAINER.format(
        trait_level=trait_level,
        trait_name=trait,
        trait_description=trait_desc,
    )
    user_prompt = USR_PROMPT_EXPLAINER.format(
        user_text=user_text,
        trait_level=trait_level,
        trait_name=trait,
        sim_text=sim_text,
    )
    return sys_prompt, user_prompt


def _build_judge_prompts(text: str, explain_1: str, explain_2: str):
    sys_prompt = SYS_PROMPT_JUDGE
    user_prompt = USR_PROMPT_JUDGE.format(
        text=text,
        explain_1=explain_1,
        explain_2=explain_2,
    )
    return sys_prompt, user_prompt


def _parse_judge_output(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw_lower = raw.lower()
    if "final judgement" in raw_lower:
        for line in raw.splitlines():
            l = line.strip().lower()
            if l.startswith("-") or l.startswith("(") or l.startswith("high") or l.startswith("low"):
                if "high" in l:
                    return "high"
                if "low" in l:
                    return "low"
    if "high" in raw_lower:
        return "high"
    if "low" in raw_lower:
        return "low"
    return None


async def _run_explainer(
    trait: str,
    trait_level: str,
    user_text: str,
    sim_text: str,
    model_name: str,
    log_path: Optional[str],
    record_idx: int,
):
    sys_prompt, user_prompt = _build_explainer_prompts(trait, trait_level, user_text, sim_text)
    return await _call_async(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        model_name=model_name,
        temperature=EXPLAINER_TEMPERATURE,
        max_new_tokens=EXPLAINER_MAX_TOKENS,
        top_p=EXPLAINER_TOP_P,
        log_filepath=log_path,
        record_idx=record_idx,
        trait_col=f"explainer_{trait_level}",
    )


async def _run_judge(
    text: str,
    explain_1: str,
    explain_2: str,
    model_name: str,
    log_path: Optional[str],
    record_idx: int,
    trait: str,
):
    sys_prompt, user_prompt = _build_judge_prompts(text, explain_1, explain_2)
    return await _call_async(
        user_prompt=user_prompt,
        system_prompt=sys_prompt,
        model_name=model_name,
        temperature=JUDGE_TEMPERATURE,
        max_new_tokens=JUDGE_MAX_TOKENS,
        top_p=JUDGE_TOP_P,
        log_filepath=log_path,
        record_idx=record_idx,
        trait_col=f"judge_{trait}",
    )


async def process_record(
    idx: int,
    row: pd.Series,
    retriever: RAGRetriever,
    trait: str,
    model_name: str,
    log_dir: Optional[str],
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        text = str(row["text"])
        trait_col = next((k for k, v in TRAIT_MAP.items() if v == trait), trait)
        true_label = str(row.get(trait_col, ""))

        log_path = os.path.join(log_dir, f"model_x_{trait}.log") if log_dir else None

        sim_context = retriever.build_explainer_context(text, trait, top_k=3)

        explain_high = await _run_explainer(trait, "high", text, sim_context, model_name, log_path, int(idx))
        explain_low = await _run_explainer(trait, "low", text, sim_context, model_name, log_path, int(idx))

        judge_raw = await _run_judge(text, explain_high, explain_low, model_name, log_path, int(idx), trait)
        pred = _parse_judge_output(judge_raw)

        return {
            "record_idx": int(idx),
            "text": text[:200],
            "true_label": true_label,
            "pred_label": pred,
            "explain_high": explain_high,
            "explain_low": explain_low,
            "judge_raw": judge_raw,
        }


async def run_batch(
    df: pd.DataFrame,
    trait: str,
    model_name: str,
    db_dir: str,
    log_dir: Optional[str],
    max_concurrency: int = 10,
):
    retriever = RAGRetriever(db_dir=db_dir)
    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = []
    for idx, row in df.iterrows():
        tasks.append(
            process_record(
                idx, row, retriever, trait, model_name, log_dir, semaphore
            )
        )

    from tqdm.asyncio import tqdm_asyncio
    results = await tqdm_asyncio.gather(*tasks)
    return pd.DataFrame(results)


async def run_all_traits(
    df: pd.DataFrame,
    model_name: str,
    db_dir: str,
    log_dir: Optional[str],
    max_concurrency: int = 10,
):
    all_results = {}
    for trait in TRAIT_MAP.values():
        print(f"\n--- Processing trait: {trait} ---")
        trait_df = await run_batch(
            df, trait, model_name, db_dir, log_dir, max_concurrency
        )
        trait_results = {}
        for _, row in trait_df.iterrows():
            trait_results[int(row["record_idx"])] = row["pred_label"]
        all_results[trait] = trait_results
    return all_results
