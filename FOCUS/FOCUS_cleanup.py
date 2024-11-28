import csv
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def classify_sentence(sentence:str) -> str:
    prompt = f"what is the conclusion of the following prediction is it Offensive or Non-offensive only provide one of these responses. Here is the sentnce '{sentence}'"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    rephrased_text = response.choices[0].message.content.strip()
    return rephrased_text

def process_csv(input_file: str, output_file: str) -> None:
    stage_1_counter  = 0
    stage_2_counter  = 0
    stage_3_counter  = 0
    stage_4_counter  = 0
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()  

            for row in reader:
                stage_1 = row.get('Stage 1', '').strip()
                stage_2 = row.get('Stage 2', '').strip()
                stage_3 = row.get('Stage 3', '').strip()
                stage_4 = row.get('Stage 4', '').strip()
                
                
                if len(stage_1.split()) > 1: 
                    stage_1_counter += 1
                    classification = classify_sentence(stage_1)
                    row['Stage 1'] = classification 
                if len(stage_2.split()) > 1: 
                    stage_2_counter += 1
                    classification = classify_sentence(stage_2)
                    row['Stage 2'] = classification 
                if len(stage_3.split()) > 1: 
                    stage_3_counter += 1
                    classification = classify_sentence(stage_3)
                    row['Stage 3'] = classification 
                if len(stage_4.split()) > 1: 
                    stage_4_counter += 1
                    classification = classify_sentence(stage_4)
                    row['Stage 4'] = classification 
                writer.writerow(row)
    print(f"Updated in stage 1: {stage_1_counter}")
    print(f"Updated in stage 2: {stage_1_counter}")
    print(f"Updated in stage 3: {stage_1_counter}")
    print(f"Updated in stage 4: {stage_1_counter}")

if __name__ == '__main__':
    input_file = '../data/Dataset_nlp_project_FOCUS_no_context_unclear_removed_3.5.csv'
    output_file = '../data/Dataset_nlp_project_FOCUS_cleaned.csv'
    process_csv(input_file, output_file)
    print(f"Processed CSV saved to: {output_file}")
