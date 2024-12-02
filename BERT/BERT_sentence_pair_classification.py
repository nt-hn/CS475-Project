import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score

class OffensiveWordDataset(Dataset):
    def __init__(self, misspelled_words, contexts, labels, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.misspelled_words = misspelled_words
        self.contexts = contexts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.misspelled_words[idx],
            self.contexts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class OffensiveWordClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    report = classification_report(all_labels, all_predictions, target_names=['Not Offensive', 'Offensive'])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detailed_report': report
    }

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        
        # Evaluation phase
        val_metrics = evaluate_model(model, val_loader, device)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Metrics:')
        print(f'Accuracy: {val_metrics["accuracy"]*100:.2f}%')
        print(f'Precision: {val_metrics["precision"]:.4f}')
        print(f'Recall: {val_metrics["recall"]:.4f}')
        print(f'F1 Score: {val_metrics["f1"]:.4f}')
        print('\nDetailed Classification Report:')
        print(val_metrics['detailed_report'])
        
        # Save best model based on both F1 score and accuracy
        if val_metrics['f1'] > best_f1 or (val_metrics['f1'] == best_f1 and val_metrics['accuracy'] > best_accuracy):
            best_f1 = val_metrics['f1']
            best_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(), 'BERT.pth')
            print(f'New best model saved! F1: {best_f1:.4f}, Accuracy: {best_accuracy*100:.2f}%')

def main():
    # Load and preprocess data
    df = pd.read_csv('./data/Dataset_nlp_project_BIO.csv')
    
    label_map = {'O': 1, 'N': 0}
    df['Label (O & N)'] = df['Label (O & N)'].map(label_map)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label (O & N)'])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    train_dataset = OffensiveWordDataset(
        train_df['Misspelled Word'].tolist(),
        train_df['Context'].tolist(),
        train_df['Label (O & N)'].tolist(),
        tokenizer
    )
    
    val_dataset = OffensiveWordDataset(
        val_df['Misspelled Word'].tolist(),
        val_df['Context'].tolist(),
        val_df['Label (O & N)'].tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OffensiveWordClassifier(bert_model).to(device)
    
    train_model(model, train_loader, val_loader, device)

    print("\nFinal Model Evaluation:")
    final_metrics = evaluate_model(model, val_loader, device)
    print(f"Final Accuracy: {final_metrics['accuracy']*100:.2f}%")
    print(final_metrics['detailed_report'])

if __name__ == "__main__":
    main()
