from src.baselines.prompts import ZEROSHOT_SYS_PROMPT


def build_system_prompt(trait_name: str):
    return ZEROSHOT_SYS_PROMPT.format(trait=trait_name)
