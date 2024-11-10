import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics(df):
    # Convert 'Label' column to actual labels (1 for 'O' and 0 for 'N')
    actual_labels = df['Label (O & N)'].apply(lambda x: 1 if x == 'O' else 0).tolist()
    
    # Columns for stages to be evaluated
    stages = ['Stage 1']
    
    # Store results for each stage
    results = {}
    
    for stage in stages:
        # Convert predicted labels with 1 for offensive and 0 for non-offensive, None for invalid entries
        predicted_labels = df[stage].apply(lambda x: 1 if x in ['Offensive', 'Offensive.'] else 0 if x in ['Non-offensive', 'Non-offensive.'] else None)
        
        # Filter out None values by dropping corresponding rows in both actual and predicted labels
        valid_data = pd.DataFrame({'actual': actual_labels, 'predicted': predicted_labels}).dropna()
        filtered_actual_labels = valid_data['actual'].tolist()
        filtered_predicted_labels = valid_data['predicted'].tolist()
        
        # Calculate metrics if there are valid predictions
        if filtered_predicted_labels:
            accuracy = accuracy_score(filtered_actual_labels, filtered_predicted_labels)
            precision = precision_score(filtered_actual_labels, filtered_predicted_labels, zero_division=0)
            recall = recall_score(filtered_actual_labels, filtered_predicted_labels, zero_division=0)
            f1 = f1_score(filtered_actual_labels, filtered_predicted_labels, zero_division=0)
            
            # Store the metrics in the dictionary
            results[stage] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
        else:
            # If no valid predictions, set metrics to None
            results[stage] = {
                'Accuracy': None,
                'Precision': None,
                'Recall': None,
                'F1 Score': None
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
            print(f"{metric}: {value if value is not None else 'N/A'}")
        print()

# Specify the path to the CSV file and process it
csv_file_path = './data/Dataset_nlp_project_FOCUS_unclear_removed_3.5.csv'
process_csv(csv_file_path)