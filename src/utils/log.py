from typing import Union, List
import asyncio
import os


_lock = asyncio.Lock()


def log_to_file(
    log_filepath: str,
    system_prompt: str,
    user_prompt: Union[str, List[str]],
    response: Union[str, List[str]],
    record_idx: int = None,
    trait_col: str = None,
):
    if not log_filepath:
        return

    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    header = ""
    if record_idx is not None or trait_col is not None:
        header = f"[Record {record_idx}, Trait {trait_col}]\n"

    with open(log_filepath, "a", encoding="utf-8") as f:
        if header:
            f.write(header)
        f.write(f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n")
        f.write(f"--- USER PROMPT ---\n{user_prompt}\n")
        f.write("\n")
        response_str = str(response)
        f.write(f"--- LLM RESPONSE ---\n{response_str}\n")
        f.write("=" * 80 + "\n\n")


async def log_to_file_async(
    log_filepath: str,
    system_prompt: str,
    user_prompt: Union[str, List[str]],
    response: Union[str, List[str]],
    record_idx: int = None,
    trait_col: str = None,
):
    if not log_filepath:
        return

    header = ""
    if record_idx is not None or trait_col is not None:
        header = f"[Record {record_idx}, Trait {trait_col}]\n"

    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

    async with _lock:
        with open(log_filepath, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(f"--- SYSTEM PROMPT ---\n{system_prompt}\n\n")
            f.write(f"--- USER PROMPT ---\n{user_prompt}\n")
            f.write("\n")
            response_str = str(response)
            f.write(f"--- LLM RESPONSE ---\n{response_str}\n")
            f.write("=" * 80 + "\n\n")
