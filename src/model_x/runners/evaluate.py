import asyncio
import os
import time

import pandas as pd

from .async_runner import run_all_traits
from ...common.reporters import write_classification_report
from ...common.trait_defs import TRAIT_COLS, TRAIT_MAP


def run(
    test_csv: str,
    model_name: str,
    db_dir: str = "data/vector_db/essays",
    output_base: str = None,
    log_base: str = None,
    max_concurrency: int = 10,
):
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if output_base is None:
        output_base = os.path.join(_root, "experiment", model_name)
    if log_base is None:
        log_base = os.path.join(_root, "log", model_name, "model_x")

    os.makedirs(output_base, exist_ok=True)
    os.makedirs(log_base, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_base, run_id)
    log_dir = os.path.join(log_base, run_id)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    df = pd.read_csv(test_csv)
    n_records = len(df)
    print(f"[{run_id}] Loaded {n_records} records from {test_csv}")
    print(f"Model: {model_name}, DB: {db_dir}, Max concurrency: {max_concurrency}")

    t0 = time.time()
    all_results = asyncio.run(
        run_all_traits(df, model_name, db_dir, log_dir, max_concurrency)
    )
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / n_records:.2f}s/record)")

    rows = []
    for idx in range(n_records):
        row_out = {}
        for trait_name, trait_results in all_results.items():
            trait_col = next((k for k, v in TRAIT_MAP.items() if v == trait_name), trait_name)
            row_out[f"pred_{trait_col}"] = trait_results.get(idx)
        rows.append(row_out)

    pred_df = pd.DataFrame(rows)
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
        fail_count=0,
        prompt_mode="model_x",
        top_k=3,
        vector_db_dir=db_dir,
        max_concurrency=max_concurrency,
        elapsed=elapsed,
    )
    print(f"Classification report saved to {report_path}")
    return save_df, report_path