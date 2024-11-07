import csv

def check_masked_phrase_in_column(file_path:str) -> int:
    counter = 0
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            sample_value = row.get('Replaced entities', '')
            if '[MASKED_PHRASE]' not in sample_value:
                print(row)
                counter += 1
    
    return counter

def check_rephrase_and_original_identical(file_path: str) -> int:
    counter = 0
    with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            misspelled_word = row.get('Misspelled Word', '')
            context = row.get('Context', '').replace(misspelled_word, '[MASKED_PHRASE]').lower().strip()
            rephrased_word = row.get('Replaced entities', '').lower().strip()
            
            if rephrased_word == context:
                print(row)
                counter += 1
    return counter

file_path = './data/Dataset_nlp_project_rephrased.csv'  

if __name__ == '__main__':
    availability_counter = check_masked_phrase_in_column(file_path)
    similarity_counter = check_rephrase_and_original_identical(file_path)

print(f"Number of rows without '[masked phrase]': {availability_counter}")
print(f"Number of rows identical: {similarity_counter}")
