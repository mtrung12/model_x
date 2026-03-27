from src.clients.llama_client import llama_call


def call_sync(user_prompt: str, system_prompt: str, model_name: str, max_new_tokens: int, log_filepath: str):
    return llama_call(user_prompt, system_prompt, model_name, max_new_tokens, log_filepath)
