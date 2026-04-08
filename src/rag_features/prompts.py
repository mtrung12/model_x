
import random


TRAITS = {
    "Neuroticism": {
        "high": "High Neuroticism scorers are at risk of psychiatric problems, prone to irrational ideas, less able to control impulses, and cope poorly with stress.",
        "low": "Low Neuroticism scorers are emotionally stable, calm, even-tempered, relaxed, and able to face stressful situations without becoming upset.",
    },
    "Extraversion": {
        "high": "Extraverts are energetic and optimistic.",
        "low": "Introverts are reserved rather than unfriendly, independent rather than followers, even-paced rather than sluggish.",
    },
    "Openness to Experience": {
        "high": "People scoring high on Openness tend to be unconventional, willing to question authority and prepared to entertain new ethical, social and political ideas. They are curious about both inner and outer worlds, experientially richer, willing to entertain novel ideas and unconventional values, and experience both positive and negative emotions more keenly.",
        "low": "People scoring low on Openness tend to be conventional in behaviour and conservative in outlook. They prefer the familiar to the novel, and their emotional responses are somewhat muted.",
    },
    "Agreeableness": {
        "high": "An agreeable person is fundamentally altruistic, sympathetic to others and eager to help them, and believes that others will be equally helpful.",
        "low": "The disagreeable or antagonistic person is egocentric, sceptical of others' intentions, and competitive rather than co-operative.",
    },
    "Conscientiousness": {
        "high": "The conscientious person is purposeful, strong-willed and determined. Conscientiousness is manifested in achievement orientation (hardworking and persistent), dependability (responsible and careful) and orderliness (planful and organised). On the negative side, high Conscientiousness may lead to annoying fastidiousness, compulsive neatness or workaholic behaviour.",
        "low": "Low scorers may not necessarily lack moral principles, but they are less exacting in applying them.",
    },
}


SYS_TEMPLATE = """You are a psychological analyst specializing in personality assessment. Your task is to extract behavioral features indicative of {trait_name} from a person's text.
For a richer and more multifaceted analysis, generate explanations considering the following four psycholinguistic elements:
Emotion: Expressed through words that indicate positive or negative feelings (e.g., "happy," "ugly," "bitter"). This encompasses positive emotions, including feelings of optimism and energy (e.g., "pride," "win"), as well as negative emotions that convey anxiety or fear (e.g., "nervous," "afraid"), anger (e.g., "hate," "kill"), and sadness or depression (e.g., "grief," "cry").
Cognition: Represented by words related to active cognitive mechanisms and thinking strategies (e.g., "cause," "know," "ought"). This includes terms that indicate causation (e.g., "because," "effect"), insight (e.g., "think," "consider"), discrepancy (e.g., "should," "would"), inhibition (e.g., "block," "constrain"), tentativeness (e.g., "maybe," "perhaps"), and certainty (e.g., "always," "never").
Sensory Perception: Reflected by words associated with the physical senses (e.g., "see," "touch," "listen"). This captures the acts of seeing (e.g., "view," "look"), hearing (e.g., "heard," "sound"), and feeling (e.g., "touch," "felt").
Sociality: Indicated by words reflecting interpersonal interactions and relationships with others (e.g., "talk," "us," "friend"). This includes words specifically related to communication (e.g., "share," "converse"), friends (e.g., "pal," "buddy"), family (e.g., "mom," "brother"), humans in general (e.g., "boy," "group"), and other general references to people, which captures the use of first-person plural, second-person, and third-person pronouns.

Provide a rich, descriptive assessment (3-5 sentences) that a downstream model can use to determine if this person is high or low in {trait_name}. Ground your assessment only in what is explicitly present in the text. Avoid speculation.
{label_high} {trait_name}: {trait_high}
{label_low} {trait_name}: {trait_low}

Output format:
Emotion: - explanation
Cognition: - explanation
Sensory perception: - explanation
Sociality: - explanation
"""

USER_TEMPLATE = """Analyze the following text and extract features indicative of {trait_name}:

---
{text}
---

Output your assessment covering behavioral patterns, emotional cues, and social indicators that suggest {trait_name} levels. If insufficient evidence exists, state "Limited signal detected."
"""


def get_extraction_prompts(trait_key: str) -> tuple[str, str]:
    """Return (sys_prompt, user_prompt) for feature extraction on a given trait."""
    t = TRAITS[trait_key]
    if random.random() < 0.5:
        trait_1, trait_2 = t["high"], t["low"]
        label_1, label_2 = "High", "Low"
    else:
        trait_1, trait_2 = t["low"], t["high"]
        label_1, label_2 = "Low", "High"
    sys_prompt = SYS_TEMPLATE.format(
        trait_name=trait_key,
        trait_high=trait_1,
        trait_low=trait_2,
        label_high=label_1,
        label_low=label_2,
    )
    user_prompt = USER_TEMPLATE.format(
        trait_name=trait_key,
        text="{text}",
    )
    return sys_prompt, user_prompt


def build_extraction_messages(trait_key: str, text: str) -> list[dict]:
    """Build a ChatML-format messages list for feature extraction."""
    sys_prompt, user_template = get_extraction_prompts(trait_key)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_template.format(text=text)},
    ]
