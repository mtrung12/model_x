import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from src.utils.log import log_to_file, log_to_file_async
from dotenv import load_dotenv

load_dotenv()

_pipe_cache = {}


def get_HF_pipeline(model_name: str, max_new_tokens: int = 512):
    if model_name in _pipe_cache:
        return _pipe_cache[model_name]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = torch.compile(model, mode="reduce-overhead")

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "do_sample": False,
        "return_full_text": False,
    }
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_config)
    _pipe_cache[model_name] = pipe
    return pipe


def create_message_HF(system_prompt_str: str, user_prompt_str: str):
    return [
        {"role": "system", "content": system_prompt_str},
        {"role": "user", "content": user_prompt_str},
    ]


def clear_pipe_cache(model_name: str = None):
    global _pipe_cache
    if model_name is None:
        _pipe_cache.clear()
    else:
        if model_name in _pipe_cache:
            del _pipe_cache[model_name]
    gc.collect()
    torch.cuda.empty_cache()


def llama_call(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    max_new_tokens: int = 512,
    log_filepath: str = None,
    clear_cache_after: bool = False,
):
    pipe = get_HF_pipeline(model_name, max_new_tokens)
    message = create_message_HF(system_prompt, user_prompt)
    prompt = pipe.tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(prompt)
    result = outputs[0]["generated_text"]
    if log_filepath:
        log_to_file(log_filepath, system_prompt, user_prompt, result)
    del outputs
    if clear_cache_after:
        clear_pipe_cache(model_name)
    return result


async def llama_call_async(
    user_prompt: str,
    system_prompt: str,
    model_name: str,
    max_new_tokens: int = 512,
    log_filepath: str = None,
    record_idx: int = None,
    trait_col: str = None,
    clear_cache_after: bool = False,
):
    import asyncio

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, llama_call, user_prompt, system_prompt, model_name, max_new_tokens, None, clear_cache_after
    )
    if log_filepath:
        await log_to_file_async(log_filepath, system_prompt, user_prompt, result, record_idx=record_idx, trait_col=trait_col)
    return result
