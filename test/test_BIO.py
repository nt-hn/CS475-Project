import pandas as pd

def verify_BIO_data(input_csv):
    df = pd.read_csv(input_csv)

    for index, row in df.iterrows():
        context = row['Context']
        labels = eval(row['Output Labels'])  
        
        context_tokens = context.split()  
        
        if len(context_tokens) != len(labels):
            print(f"Error at row {index}: The number of context words ({len(context_tokens)}) does not match the number of labels ({len(labels)})")
        if 'B-O' not in labels and 'B-N' not in labels:
            print(f"Error at row {index}: There is no 'B-O' or 'B-N' label in the output labels.")
        

if __name__ == '__main__':
    input_csv = '../data/Dataset_nlp_project_BIO.csv'
    verify_BIO_data(input_csv)
