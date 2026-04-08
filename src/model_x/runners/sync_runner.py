import gc
import os
import time
import random
import pandas as pd
import torch

from ..prompts import (
    TRAITS,
    SYS_PROMPT_EXPLAINER,
    USR_PROMPT_EXPLAINER,
    SYS_PROMPT_JUDGE,
    USR_PROMPT_JUDGE,
)
from ...common.trait_defs import TRAIT_COLS, TRAIT_MAP
from ...common.reporters import write_classification_report
from ...clients.llama_client import llama_call, clear_pipe_cache
from ..retriever import RAGRetriever


EXPLAINER_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 512


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


def _parse_judge_output(raw: str):
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


def run_llama(
    test_csv: str,
    model_name: str,
    db_dir: str = "data/vector_db/essays",
    output_dir: str = None,
    log_filepath: str = None,
    train_csv: str = None,
):
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "experiment",
            model_name,
        )
    os.makedirs(output_dir, exist_ok=True)

    if log_filepath is None:
        log_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "log",
            model_name,
            "model_x",
        )
        os.makedirs(log_base, exist_ok=True)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        log_filepath = os.path.join(log_base, f"{run_id}.log")
    else:
        run_id = os.path.splitext(os.path.basename(log_filepath))[0]

    df = pd.read_csv(test_csv)
    n_records = len(df)
    print(f"[{run_id}] Loaded {n_records} records from {test_csv}")
    print(f"Running llama model_x (sequential)")

    retriever = RAGRetriever(db_dir=db_dir)

    t0 = time.time()
    results_list = []
    fail_count = 0

    for idx in range(n_records):
        row = df.iloc[idx]
        text = str(row["text"])
        record_results = {}
        errors = []

        for trait in TRAIT_MAP.values():
            try:
                sim_context = retriever.build_explainer_context(text, trait, top_k=3)

                explain_high_sys, explain_high_usr = _build_explainer_prompts(
                    trait, "high", text, sim_context
                )
                explain_high = llama_call(
                    explain_high_usr,
                    explain_high_sys,
                    model_name,
                    max_new_tokens=EXPLAINER_MAX_TOKENS,
                    log_filepath=log_filepath,
                )

                explain_low_sys, explain_low_usr = _build_explainer_prompts(
                    trait, "low", text, sim_context
                )
                explain_low = llama_call(
                    explain_low_usr,
                    explain_low_sys,
                    model_name,
                    max_new_tokens=EXPLAINER_MAX_TOKENS,
                    log_filepath=log_filepath,
                )

                if random.random() < 0.5:
                    explain_1, explain_2 = explain_high, explain_low
                else:
                    explain_1, explain_2 = explain_low, explain_high

                judge_sys, judge_usr = _build_judge_prompts(text, explain_1, explain_2)
                judge_raw = llama_call(
                    judge_usr,
                    judge_sys,
                    model_name,
                    max_new_tokens=JUDGE_MAX_TOKENS,
                    log_filepath=log_filepath,
                )
                pred = _parse_judge_output(judge_raw)
                record_results[trait] = pred
            except Exception as e:
                errors.append(f"{trait}: {e}")
                record_results[trait] = None

        # Aggressive memory cleanup between records to prevent OOM
        clear_pipe_cache(model_name)
        gc.collect()
        torch.cuda.empty_cache()

        if errors:
            fail_count += 1
            print(f"Record {idx}: errors — {'; '.join(errors)}")
        else:
            print(f"Record {idx}: done")

        results_list.append(record_results)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / n_records:.2f}s/record). {fail_count} record(s) had errors.")

    pred_rows = []
    for results in results_list:
        row = {}
        for col in TRAIT_COLS:
            trait_name = next((k for k, v in TRAIT_MAP.items() if v == col), col)
            row[f"pred_{col}"] = results.get(trait_name) if results else None
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)

    save_df = pd.concat([df[["text"] + TRAIT_COLS].reset_index(drop=True), pred_df], axis=1)

    raw_out_path = os.path.join(output_dir, "raw_predictions.csv")
    save_df.to_csv(raw_out_path, index=False)
    print(f"Raw predictions saved to {raw_out_path}")

    report_path = os.path.join(output_dir, "classification_report.txt")
    write_classification_report(
        report_path=report_path,
        save_df=save_df,
        trait_cols=TRAIT_COLS,
        trait_map=TRAIT_MAP,
        run_id=run_id,
        model_name=model_name,
        test_csv=test_csv,
        n_records=n_records,
        fail_count=fail_count,
        prompt_mode="model_x",
        top_k=3,
        vector_db_dir=db_dir,
        max_concurrency=1,
        elapsed=elapsed,
    )

    print(f"Classification report saved to {report_path}")
    return save_df, report_path
