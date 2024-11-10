paraphrase_prompt = """
    Word: {mispelled_word}
    Context: {context}

    Task Instructions:

    1. Entity Replacement:
    - Randomly select at least half of the significant words in the sentence, such as nouns, adjectives, verbs, and key phrases, excluding the target word {mispelled_word}.
    - Replace these selected words with new entities that fit naturally in the context but introduce completely new terminology (e.g., seasonal references, actions, traits).
    - The new words should be contextually relevant but distinctly different from the original ones.
    2. Sentence Reconstruction:
    - Replace {mispelled_word} with [MASKED_PHRASE].
    - Incorporate the newly selected entities into the sentence in a way that maintains the original theme and tone while introducing the changes.
    - Ensure the sentence remains grammatically correct, logically coherent, and the new words fit naturally. 
    Example Input:
    Context: As the days shorted, she embraced the winter arc, intensifying her fitness routine amidst the growing chill, silently heralding warmer days through steadfast effort.
    Word: winter arc

    Example Output:
    As the nights lengthened, she embraced the [MASKED_PHRASE], deepening her reading habit amidst the scattering of leaves, silently heralding warmer days through steadfast effort.

    Output format: Reconstructed Sentence Only
"""

direct_prompt = "Answer with only 'Offensive' or 'Non-offensive'. Word: {mispelled_word}\nExample: {context}\n Offensive or not?:"

first_stage_prompt = """
    Task Instruction: Given a word or a phrase, use the provided context to infer if the given word or phrase is offensive or non-offensive. When analyzing the usage contexts, interpret the context literally and think through it carefully to infer if the word (or the phrase) is offensive or non-offensive. 
    Word: {mispelled_word}
    Context: {context}
    Output format: Offensive or Non-offensive
"""

second_stage_prompt = """
    Task Instruction: Given a context with a masked word (or phrase), analyze the context and if the given word or phrase is offensive or non-offensive. Think through the context carefully to deduce if the given word or phrase is offensive or non-offensive.
    Context with Masked Phrase: {context_with_masked_mispelled_word}
    Output format: Offensive or Non-offensive
"""

third_stage_prompt = """
    Task Instruction: Given a paraphrased context with a masked word (or phrase), analyze the context and if the given word or phrase is offensive or non-offensive. Think through the context carefully to deduce if the given word or phrase is offensive or non-offensive.
    Paraphrased context with Masked Phrase: {paraphrased_context_with_mask}
    Output format: Offensive or Non-offensive
"""

fourth_stage_prompt = """
    Task Instruction: Given a word or phrase and its context, determine whether the phrase is offensive or non-offensive.

    Think through the following but don't output anything yet:
    1. Direct Interpretation: Consider the meaning of the word or phrase in the given context and whether it could be offensive or non-offensive.
    2. Contextual Inference: Consider how the word or phrase might be understood if it were masked in the context, and whether it is offensive or non-offensive.
    3. Reconstructed Context Inference: Think about the word or phrase in a reconstructed context, where some related entities are changed, and determine if it's offensive or non-offensive.

    Based on these thoughts, output your final judgment as one of the following labels:

    Word: {mispelled_word}
    Context: {context}

    Output: Offensive or  Non-offensive
"""
