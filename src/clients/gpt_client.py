from openai import OpenAI
from dotenv import load_dotenv
import os
from src.utils.log import log_to_file

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass


load_dotenv()

_client = None
_async_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def get_async_client():
    global _async_client
    if _async_client is None:
        from openai import AsyncOpenAI

        _async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _async_client


def create_message_openai(system_prompt_str, user_prompt_str):
    return [
        {"role": "system", "content": system_prompt_str},
        {"role": "user", "content": user_prompt_str},
    ]


def gpt_call(
    user_prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    log_filepath: str,
):
    client = get_client()
    params = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_new_tokens is not None:
        params["max_tokens"] = max_new_tokens

    params["messages"] = create_message_openai(system_prompt, user_prompt)

    content = client.chat.completions.create(**params).choices[0].message.content
    log_to_file(log_filepath, system_prompt, user_prompt, content)
    return content


async def gpt_call_async(
    user_prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    log_filepath: str,
    record_idx: int = None,
    trait_col: str = None,
):
    client = get_async_client()
    params = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_new_tokens is not None:
        params["max_tokens"] = max_new_tokens

    params["messages"] = create_message_openai(system_prompt, user_prompt)

    response = await client.chat.completions.create(**params)
    content = response.choices[0].message.content
    await log_to_file_async(log_filepath, system_prompt, user_prompt, content, record_idx=record_idx, trait_col=trait_col)
    return content
