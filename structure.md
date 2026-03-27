src/
  common/                         # ALL shared code
    parsers.py                   # parse_llm_output()
    reporters.py                  # write_classification_report()
    trait_defs.py                # TRAIT_MAP, TRAIT_COLS, etc.
    prompts/
      baseline.py               # ZEROSHOT, ONESHOT, COT, ...
      pipeline.py                # FEATURE_EXTRACTION, PERSONALITY_PREDICTION, ...

  clients/                        # Raw LLM calls — shared across baselines AND pipeline
    gpt_client.py                # gpt_call (sync) + gpt_call_async (async)
    llama_client.py               # llama_call (sync only) ← used by BASELINES (sync runner)
                                 #                         AND PIPELINE (when pipeline
                                 #                         itself runs sequentially for Llama)

  baselines/                      # oneshot / zeroshot / cot — Llama: sync runner
    runners/
      sync_runner.py              # for Llama: loop over records + traits, one-at-a-time
      async_runner.py             # for GPT: asyncio.Semaphore + gather
    modes/
      zeroshot.py
      oneshot.py
      cot.py
    evaluate.py

  model_x/                       # Advanced RAG pipeline 
    steps/
      extract_features.py         # llama_call_async OR gpt_call_async
      retrieve.py                  # retrieve_similar (sync, from rag/)
      predict_all.py
    run.py
    evaluate.py

  rag/
    create_vector_db.py
    retrieve_similar.py
