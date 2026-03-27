from src.common.prompts.baseline import COT_SYS_PROMPT


def build_system_prompt(trait_name: str):
    return COT_SYS_PROMPT.format(trait=trait_name)
