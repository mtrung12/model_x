ZEROSHOT_SYS_PROMPT = """Based on a human-generated text, predict whether the person’s perspective of {trait} (one of the Big Five personality traits) is ‘high’ or ‘low’.

Output format:
Prediction
- ‘high’ or ‘low’"""

ONESHOT_SYS_PROMPT = """Based on a human-generated text, predict whether the person’s perspective of {trait} (one of the Big Five personality traits) is ‘high’ or ‘low’.

Example:
Text: {example_text}
Prediction
- {example_label}

Output format:
Prediction
- ‘high’ or ‘low’"""

COT_SYS_PROMPT = """Based on a human-generated text, predict whether the person's perspective of {trait} (one of the Big Five personality traits) is ‘high’ or ‘low’.
Let's think step-by-step as follows:
1. Identify specific phrases, behaviors, or patterns that indicate high Conscientiousness (e.g., planning, organization, responsibility, attention to detail, perseverance) or low Conscientiousness (e.g., disorganization, procrastination, impulsivity, carelessness).
2. Weigh the evidence and explain your reasoning in detail.
3. Only after your full analysis, give the final prediction.

Output format:
Prediction
- ‘high’ or ‘low’
"""

NORMAL_USER_PROMPT = """Text: {text}"""
