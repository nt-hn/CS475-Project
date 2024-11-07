import os
import csv
from dotenv import load_dotenv
from openai import OpenAI
from prompts import paraphrase_prompt

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

def rephrase_sentence(context:str, mispelled_word:str) -> str:
    prompt = paraphrase_prompt.format(mispelled_word=mispelled_word, context=context)
    response = client.chat.completions.create(
        model="gpt-4-turbo",
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

def process_csv(input_file:str, output_file:str) -> None:
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['Replaced entities']

        with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                context = row['Context']
                misspelled_word = row['Misspelled Word']
                rephrased_text = rephrase_sentence(context, misspelled_word)
                row['Replaced entities'] = rephrased_text
                writer.writerow(row)

    print(f"Processed file saved to {output_file}")

input_file = './data/Dataset_nlp_project.csv'  
output_file = './data/Dataset_nlp_project_rephrased.csv'  

if __name__ == '__main__':
    process_csv(input_file, output_file)
    print('Done!')
