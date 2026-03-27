import asyncio
import os
import time

import pandas as pd

from src.baselines.runners.async_runner import run_batch
from src.common.reporters import write_classification_report
from src.common.trait_defs import TRAIT_COLS, TRAIT_MAP


def run_parallel(
    test_csv: str,
    model_name: str,
    max_new_tokens: int,
    prompt_mode: str = "zeroshot",
    top_k: int = 5,
    vector_db_dir: str = "data/vector_db",
    output_dir: str = None,
    max_concurrency: int = 10,
    train_csv: str = None,
    log_filepath: str = None,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment", model_name)
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
    print(f"Running with prompt_mode={prompt_mode}, max_concurrency={max_concurrency}")

    t0 = time.time()
    results_list, fail_count = asyncio.run(
        run_batch(
            df,
            model_name,
            max_new_tokens,
            log_filepath,
            prompt_mode,
            top_k,
            vector_db_dir,
            max_concurrency,
            train_csv=train_csv,
        )
    )
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
        top_k=top_k,
        vector_db_dir=vector_db_dir,
        max_concurrency=max_concurrency,
        elapsed=elapsed,
    )

    print(f"Classification report saved to {report_path}")
    return save_df, report_path
