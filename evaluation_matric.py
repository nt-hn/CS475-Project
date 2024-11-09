import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics(df):
    # Convert 'Label' column to a list of actual labels
    actual_labels = df['Label (O & N)'].apply(lambda x: 1 if x == 'O' else 0).tolist()  # O=1, N=0
    
    # Columns for stages to be evaluated
    stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
    
    # Store results for each stage
    results = {}
    
    for stage in stages:
        # Convert predicted labels to 1 for offensive (O) and 0 for not offensive (N)
        predicted_labels = df[stage].apply(lambda x: 1 if x == 'Offensive' or x == 'Offensive.' else 0).tolist()
        
        # Calculate metrics
        accuracy = accuracy_score(actual_labels, predicted_labels)
        precision = precision_score(actual_labels, predicted_labels, zero_division=0)  # Handle zero division
        recall = recall_score(actual_labels, predicted_labels, zero_division=0)  # Handle zero division
        f1 = f1_score(actual_labels, predicted_labels, zero_division=0)  # Handle zero division
        
        # Store the metrics in the dictionary
        results[stage] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    return results

def process_csv(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Call the evaluate_metrics function to get the results
    results = evaluate_metrics(df)
    
    # Print results
    for stage, metrics in results.items():
        print(f"Metrics for {stage}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print()

csv_file_path = './data/Dataset_nlp_project_FOCUS_unclear_removed_3.5.csv'
process_csv(csv_file_path)
