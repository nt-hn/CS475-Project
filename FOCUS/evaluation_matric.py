import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_metrics(df):
    actual_labels = df['Label (O & N)'].apply(lambda x: 1 if x == 'O' else 0).tolist()
    stages = ['Stage 1']
    results = {}
    
    for stage in stages:
        predicted_labels = df[stage].apply(lambda x: 1 if x in ['Offensive', 'Offensive.'] else 0 if x in ['Non-offensive', 'Non-offensive.'] else None)
        
        valid_data = pd.DataFrame({'actual': actual_labels, 'predicted': predicted_labels}).dropna()
        filtered_actual_labels = valid_data['actual'].tolist()
        filtered_predicted_labels = valid_data['predicted'].tolist()
        
        if filtered_predicted_labels:
            accuracy = accuracy_score(filtered_actual_labels, filtered_predicted_labels)
            precision = precision_score(filtered_actual_labels, filtered_predicted_labels, zero_division=0)
            recall = recall_score(filtered_actual_labels, filtered_predicted_labels, zero_division=0)
            f1 = f1_score(filtered_actual_labels, filtered_predicted_labels, zero_division=0)
            
            results[stage] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
        else:
            results[stage] = {
                'Accuracy': None,
                'Precision': None,
                'Recall': None,
                'F1 Score': None
            }
    
    return results

def process_csv(file_path):
    df = pd.read_csv(file_path)
    
    results = evaluate_metrics(df)
    
    for stage, metrics in results.items():
        print(f"Metrics for {stage}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value if value is not None else 'N/A'}")
        print()

if __name__ == '__main__':
    csv_file_path = '../data/Dataset_nlp_project_FOCUS_unclear_removed_3.5.csv'
    process_csv(csv_file_path)
