from src.baselines.prompts import ONESHOT_SYS_PROMPT


def build_system_prompt(trait_name: str, example_text: str, example_label: str):
    return ONESHOT_SYS_PROMPT.format(
        trait=trait_name,
        example_text=example_text,
        example_label=example_label,
    )
