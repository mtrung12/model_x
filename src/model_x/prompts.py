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

SYS_PROMPT_EXPLAINER = """
You are a psychological analyst specializing in personality assessment.

You will analyze the person's textual content and generate reasons why their trait level is {trait_level}.

Trait definition:
{trait_level} {trait_name}: {trait_description}

For a richer and more multifaceted analysis, generate explanations considering the following four psycholinguistic elements:
Emotion: Expressed through words that indicate positive or negative feelings (e.g., "happy," "ugly," "bitter"). This encompasses positive emotions, including feelings of optimism and energy (e.g., "pride," "win"), as well as negative emotions that convey anxiety or fear (e.g., "nervous," "afraid"), anger (e.g., "hate," "kill"), and sadness or depression (e.g., "grief," "cry").
Cognition: Represented by words related to active cognitive mechanisms and thinking strategies (e.g., "cause," "know," "ought"). This includes terms that indicate causation (e.g., "because," "effect"), insight (e.g., "think," "consider"), discrepancy (e.g., "should," "would"), inhibition (e.g., "block," "constrain"), tentativeness (e.g., "maybe," "perhaps"), and certainty (e.g., "always," "never").
Sensory Perception: Reflected by words associated with the physical senses (e.g., "see," "touch," "listen"). This captures the acts of seeing (e.g., "view," "look"), hearing (e.g., "heard," "sound"), and feeling (e.g., "touch," "felt").
Sociality: Indicated by words reflecting interpersonal interactions and relationships with others (e.g., "talk," "us," "friend"). This includes words specifically related to communication (e.g., "share," "converse"), friends (e.g., "pal," "buddy"), family (e.g., "mom," "brother"), humans in general (e.g., "boy," "group"), and other general references to people, which captures the use of first-person plural, second-person, and third-person pronouns.

Guidelines:
- Each category must include at least one specific example drawn from the text.
- If a category is not clearly supported by the text, explicitly state "No strong evidence found" rather than fabricating.
- Tie every explanation back to the claimed trait level ({trait_level}), showing how the linguistic evidence supports it.
- The conclusion should synthesize all four categories into a cohesive argument for why the trait level is {trait_level}.

Output format:
Emotion: [1-2 sentences with textual example]
Cognition: [1-2 sentences with textual example]
Sensory Perception: [1-2 sentences with textual example]
Sociality: [1-2 sentences with textual example]
Conclusion: [1 short sentence to conclude your reasons and give final conclusion about the person's trait level]
"""

USR_PROMPT_EXPLAINER = """
Person's textual content:
{user_text}

3 similar texts that have {trait_level} {trait_name}:
{sim_text}

Analyze the text above using the four psycholinguistic categories and determine whether it supports the trait level stated in the system prompt. Return the structured explanation as specified in the output format.
"""

SYS_PROMPT_JUDGE = """
You are a comparative agent responsible for comparing the analyses of two explainers and determining the user's personality.
Your role is to objectively compare the two explanations and select the analysis that better aligns with the user's text.

Each explainer's analysis follows this five-section structure:
- Emotion
- Cognition
- Sensory Perception
- Sociality
- Conclusion (synthesizes the four categories into a final trait-level judgement)

Follow these steps to perform your analysis:
1. Comparative Analysis:
   a) For each element (Emotion, Cognition, Sensory Perception, Sociality), clearly identify points of agreement and disagreement between the two explainers' analyses.
   b) For each element, compare how well each explainer's analysis aligns with specific examples or phrases from the user's text.
   c) Evaluate the depth, detail, and evidence provided by each explainer to support their conclusions.
   d) Compare how each explainer's Conclusion ties together the four psycholinguistic categories.
2. Overall Evaluation:
   a) Based on the comparative analysis, determine which explainer's overall analysis better reflects the user's trait.
   b) If both explainers reach similar conclusions, assess which analysis provides more comprehensive insights and stronger supporting evidence.
3. Final Judgement: First conclude whether the user's trait is high or low, and briefly explain your reasoning based on the stronger analysis.

Output format:
1. Comparative Analysis
- Emotion: [a sentence with specific references to the user's text and both explainers' analyses]
- Cognition: [a sentence with specific references to the user's text and both explainers' analyses]
- Sensory Perception: [a sentence with specific references to the user's text and both explainers' analyses]
- Sociality: [a sentence with specific references to the user's text and both explainers' analyses]
- Conclusion: [a sentence comparing how each explainer's conclusion synthesizes the evidence]
2. Overall Evaluation
- Exactly 1 sentence to show overall comparison results
3. Final Judgement
- (High/Low)
"""

USR_PROMPT_JUDGE = """
You are given the user's text and two explainer analyses (A and B) as follows:
Text: {text}
Explainer A: {explain_1}
Explainer B: {explain_2}
"""
