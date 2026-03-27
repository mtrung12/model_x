import os
import time
import pandas as pd

from src.clients.llama_client import llama_call
from src.common.parsers import parse_llm_output
from src.common.prompts.baseline import (
    NORMAL_USER_PROMPT,
    COT_SYS_PROMPT,
    ONESHOT_SYS_PROMPT,
    ZEROSHOT_SYS_PROMPT,
)
from src.common.trait_defs import TRAIT_COLS, TRAIT_MAP
from src.common.reporters import write_classification_report


def _build_system_prompt(prompt_mode: str, trait_col: str, example_text: str = None, example_label: str = None):
    trait_name = TRAIT_MAP.get(trait_col, trait_col)

    if prompt_mode == "zeroshot":
        return ZEROSHOT_SYS_PROMPT.format(trait=trait_name)
    elif prompt_mode == "oneshot":
        if example_text is None or example_label is None:
            raise ValueError("oneshot mode requires example_text and example_label")
        return ONESHOT_SYS_PROMPT.format(
            trait=trait_name,
            example_text=example_text,
            example_label=example_label,
        )
    elif prompt_mode == "cot":
        return COT_SYS_PROMPT.format(trait=trait_name)
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def run_llama(
    test_csv: str,
    model_name: str,
    max_new_tokens: int,
    prompt_mode: str = "zeroshot",
    output_dir: str = None,
    train_csv: str = None,
    log_filepath: str = None,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment", model_name, prompt_mode)
    os.makedirs(output_dir, exist_ok=True)

    if log_filepath is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log", model_name, prompt_mode)
        os.makedirs(log_dir, exist_ok=True)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        log_filepath = os.path.join(log_dir, f"{run_id}.log")
    else:
        run_id = os.path.splitext(os.path.basename(log_filepath))[0]
    df = pd.read_csv(test_csv)
    n_records = len(df)
    print(f"[{run_id}] Loaded {n_records} records from {test_csv}")
    print(f"Running llama with prompt_mode={prompt_mode} (sequential)")

    example_text = None
    example_label = None
    if prompt_mode == "oneshot":
        if train_csv is None:
            raise ValueError("oneshot mode requires train_csv to be provided")
        train_df = pd.read_csv(train_csv)
        example_row = train_df.sample(n=1, random_state=12).iloc[0]
        example_text = example_row["text"]

    t0 = time.time()
    results_list = []
    fail_count = 0

    for idx in range(n_records):
        row = df.iloc[idx]
        text = row["text"]
        user_prompt = NORMAL_USER_PROMPT.format(text=text)

        record_results = {}
        errors = []

        for trait_col in TRAIT_COLS:
            if prompt_mode == "oneshot":
                example_label = example_row[trait_col]
            system_prompt = _build_system_prompt(prompt_mode, trait_col, example_text, example_label)

            try:
                raw_output = llama_call(
                    user_prompt,
                    system_prompt,
                    model_name,
                    max_new_tokens,
                    log_filepath=log_filepath,
                )
                result = parse_llm_output(raw_output)
                record_results[trait_col] = result
            except Exception as e:
                errors.append(f"{trait_col}: {e}")
                record_results[trait_col] = None

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
            row[f"pred_{col}"] = results.get(col) if results else None
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
        prompt_mode=prompt_mode,
        top_k=None,
        vector_db_dir=None,
        max_concurrency=1,
        elapsed=elapsed,
    )

    print(f"Classification report saved to {report_path}")
    return save_df, report_path
