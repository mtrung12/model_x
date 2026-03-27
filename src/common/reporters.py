from sklearn.metrics import classification_report


def write_classification_report(
    report_path: str,
    save_df,
    trait_cols,
    trait_map,
    run_id: str,
    model_name: str,
    test_csv: str,
    n_records: int,
    fail_count: int,
    prompt_mode: str,
    top_k: int,
    vector_db_dir: str,
    max_concurrency: int,
    elapsed: float,
):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\\n")
        f.write("Personality Trait Detection — Classification Report\\n")
        f.write(f"{'='*70}\\n\\n")
        f.write(f"Run ID           : {run_id}\\n")
        f.write(f"Model            : {model_name}\\n")
        f.write(f"Test file        : {test_csv}\\n")
        f.write(f"# Records        : {n_records}\\n")
        f.write(f"# Failed         : {fail_count}\\n")
        f.write(f"Prompt mode      : {prompt_mode}\\n")
        f.write(f"Top-k RAG        : {top_k}\\n")
        f.write(f"Vector DB        : {vector_db_dir}\\n")
        f.write(f"Max Concurrency  : {max_concurrency}\\n")
        f.write(f"Elapsed (sec)    : {elapsed:.2f}\\n")
        f.write(f"Throughput (r/s) : {n_records / elapsed:.2f}\\n")
        f.write(f"{'-'*70}\\n\\n")

        for col in trait_cols:
            pred_col = f"pred_{col}"
            mask = (
                save_df[col].notna()
                & save_df[col].apply(lambda v: isinstance(v, str) and v in ("high", "low"))
                & save_df[pred_col].notna()
                & save_df[pred_col].apply(lambda v: isinstance(v, str) and v in ("high", "low"))
            )
            gt = save_df.loc[mask, col].apply(lambda x: x.lower())
            pred = save_df.loc[mask, pred_col].apply(lambda x: x.lower())

            valid_count = mask.sum()
            report = classification_report(gt, pred, labels=["high", "low"], zero_division=0)
            f.write(f"--- {col} ({trait_map[col]}) ---\\n")
            f.write(report)
            f.write(f"\\n  Valid pairs: {valid_count}/{n_records}\\n\\n")

        f.write(f"{'='*70}\\n")
        f.write("End of Report\\n")
