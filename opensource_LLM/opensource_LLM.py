import csv
import ollama
from prompts import direct_prompt
from unclear_removed_prompts import first_stage_prompt, second_stage_prompt, third_stage_prompt, fourth_stage_prompt

def FOCUS(mispelled_word: str, context: str, context_with_masked_mispelled_word: str, paraphrased_context_with_mask: str) -> dict:
    model = 'llama3.2'
    conversation_history = [
    ]
    
    # Stage 1
    first_stage_input = first_stage_prompt.format(mispelled_word=mispelled_word, context=context)
    conversation_history.append({"role": "user", "content": first_stage_input})
    response_first_stage = ollama.chat(
        model=model,
        messages=conversation_history
    )
    first_stage_output = response_first_stage['message']['content']

    conversation_history.append({"role": "assistant", "content": first_stage_output})

    # Stage 2
    second_stage_input = second_stage_prompt.format(context_with_masked_mispelled_word=context_with_masked_mispelled_word)
    conversation_history.append({"role": "user", "content": second_stage_input})
    response_second_stage = ollama.chat(
        model=model,
        messages=conversation_history
    )
    second_stage_output = response_second_stage['message']['content']

    conversation_history.append({"role": "assistant", "content": second_stage_output})

    # Stage 3
    third_stage_input = third_stage_prompt.format(paraphrased_context_with_mask=paraphrased_context_with_mask)
    conversation_history.append({"role": "user", "content": third_stage_input})
    response_third_stage = ollama.chat(
        model=model,
        messages=conversation_history
    )
    third_stage_output = response_third_stage['message']['content']

    conversation_history.append({"role": "assistant", "content": third_stage_output})
    
    # Stage 4
    fourth_stage_input = fourth_stage_prompt.format(mispelled_word=mispelled_word, context=context, first_stage_prompt= first_stage_prompt, second_stage_prompt=second_stage_input, third_stage_prompt=third_stage_input)
    conversation_history.append({"role": "user", "content": fourth_stage_input})
    response_fourth_stage = ollama.chat(
        model=model,
        messages=conversation_history
    )
    fourth_stage_output = response_fourth_stage['message']['content']

    # Collect all results into a dictionary
    results = {
        "Stage 1": first_stage_output,
        "Stage 2": second_stage_output,
        "Stage 3": third_stage_output,
        "Stage 4": fourth_stage_output,
    }
    
    return results

def DP(mispelled_word: str, context: str) -> str:
    response = ollama.chat(
        model='llama3.2',
        messages=[{"role": "user", "content":direct_prompt.format(mispelled_word=mispelled_word, context=context)}]
    )
    return response['message']['content']

def process_csv(input_file: str, output_file: str) -> None:
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'DP']

        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:  
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                misspelled_word = row['Misspelled Word']
                context = row['Context']
                context_with_masked_mispelled_word = context.replace(misspelled_word, '[MASKED_PHRASE]')
                paraphrased_context_with_mask = row['Replaced entities']  
                
                rephrased_results = FOCUS(misspelled_word, context, context_with_masked_mispelled_word, paraphrased_context_with_mask)
                DP_result = DP(misspelled_word, context)

                row.update({
                    'Stage 1': rephrased_results['Stage 1'],
                    'Stage 2': rephrased_results['Stage 2'],
                    'Stage 3': rephrased_results['Stage 3'],
                    'Stage 4': rephrased_results['Stage 4'],
                    'DP': DP_result,
                })
                
                writer.writerow(row)

    print(f"Processed file saved to {output_file}")

if __name__ == '__main__':
    input_file = './data/Dataset_nlp_project_rephrased.csv'
    output_file = './data/Dataset_nlp_project_opensource_LLM.csv'
    process_csv(input_file, output_file)
    print('Done!')