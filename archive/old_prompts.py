paraphrase_prompt = """
Word: {mispelled_word}
Context: {context}

Tasks:
1. Random Entity Replacement:
   - Randomly choose some entities related to "{mispelled_word}" in the context.
   - Create new entities that fit in the context of "{mispelled_word}".

2. Sentence Reconstruction:
   - Insert "[MASKED_PHRASE]" in place of "{mispelled_word}".
   - Replace the chosen entities with the newly created ones.
   - Ensure grammatical and logical coherence.

Output format: Reconstructed Sentence Only
"""

first_stage_prompt = """
Instruction: Given a word or a phrase, use the provided example to infer if the given word or phrase is offensive or non-offensive. When analyzing the usage examples, interpret the context literally and think through it carefully to infer if the word (or the phrase) is offensive or non-offensive. If whether it is offensive or not cannot be inferred from the example, it is appropriate to say, "It is unclear if it is offensive."
Word: {mispelled_word}
Example: {example}
Label:
"""

second_stage_prompt = """Instruction: Given an example with a masked word (or phrase), analyze the context and if the given word or phrase is offensive or non-offensive. Think through the example carefully to deduce if the given word or phrase is offensive or non-offensive. If whether it is offensive or not cannot be inferred from the context, it is appropriate to say, "It is unclear if it is offensive."
Example with Masked Word: {example_with_masked_mispelled_word}
Label:
"""

third_stage_prompt = """Instruction: Given an example with a masked word (or phrase), analyze the context and if the given word or phrase is offensive or non-offensive. Think through the example carefully to deduce if the given word or phrase is offensive or non-offensive. If whether it is offensive or not cannot be inferred from the context, it is appropriate to say, "It is unclear if it is offensive."
Paraphrased example with Masked Word: {paraphrase_prompt}
Label:
"""

fourth_stage_prompt = """Instruction: Given a phrase, use the provided example to deduce the meaning of the phrase (or word). When reviewing usage examples, interpret the context thoroughly to infer the nuanced meaning of the phrase. Break down your reasoning into step-by-step logic to arrive at a comprehensive understanding.

Word:{mispelled_word}
Usage example:{example}

1. Direct Interpretation: This is the labelling of the word or the phrase whether it is offensive or non-offensive based on its usage in a sentence. 
   - Possible Error: Misinterpretation of context or literal meaning. 
   - Label: {first_stage_prompt}

2. Contextual Inference: This is the labelling of the word or the phrase whether it is offensive or non-offensive inferred from a sentence where the phrase is masked or implied.
   - Possible Error: Incorrect inference due to lack of context or ambiguity.
   - Label: {second_stage_prompt}

3. Reconstructed Context Inference: This is the labelling of the word or the phrase whether it is offensive or non-offensive inferred from a sentence where related entities are replaced while maintaining the same relationship with the phrase.
   - Possible Error: The new entities might not perfectly mimic the original context, leading to a skewed understanding.
   - Label: {third_stage_prompt}

Task: Synthesize a comprehensive definition or explanation of the phrase, considering the potential errors and key insights from each step. 
Output example:
Word: fugging
Usage example: I fugging hate that guy man!
Label: Offensive 

Word: fugging
Usage example: He wrapped his arms around the huge tree, fugging it. 
Label: Non-offensive

Label: 
"""