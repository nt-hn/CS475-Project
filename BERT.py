import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from gensim.models import FastText
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import ast
from tqdm import tqdm
import os

class FastTextBertTokenClassifier(nn.Module):
    def __init__(self, num_labels, bert_model="bert-base-uncased", fasttext_dim=300):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.fasttext_proj = nn.Linear(fasttext_dim, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)
        self.dropout = nn.Dropout(0.1)
        
        # Freeze BERT layers for faster training
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, fasttext_embeds):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_sequence = bert_outputs.last_hidden_state
        
        fasttext_proj = self.fasttext_proj(fasttext_embeds)
        combined = torch.cat([bert_sequence, fasttext_proj], dim=-1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

class TokenClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, ft_model, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.ft_model = ft_model
        self.max_len = max_len
        self.label_map = {'S': 0, 'B-O': 1, 'B-N': 2, 'I-O': 3, 'I-N': 4}
        
        print("Pre-computing features...")
        self.encodings = self._precompute_encodings()
        self.fasttext_embeddings = self._precompute_fasttext()
        self.label_ids = self._precompute_labels()
        print("Features pre-computed.")
    
    def _precompute_encodings(self):
        return self.tokenizer(
            self.texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
    
    def _precompute_fasttext(self):
        all_embeddings = []
        for text in tqdm(self.texts, desc="Computing FastText embeddings"):
            words = text.split()
            embeddings = np.array([self.ft_model.wv[word] for word in words])
            pad_length = self.max_len - len(embeddings)
            if pad_length > 0:
                padding = np.zeros((pad_length, self.ft_model.vector_size))
                embeddings = np.vstack([embeddings, padding])
            all_embeddings.append(embeddings[:self.max_len])
        return torch.tensor(np.array(all_embeddings), dtype=torch.float32)
    
    def _precompute_labels(self):
        all_labels = []
        for label in self.labels:
            label_ids = [self.label_map[l] for l in label]
            pad_length = self.max_len - len(label_ids)
            if pad_length > 0:
                label_ids.extend([-100] * pad_length)
            all_labels.append(label_ids[:self.max_len])
        return torch.tensor(all_labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'fasttext_embeds': self.fasttext_embeddings[idx],
            'labels': self.label_ids[idx]
        }

@torch.no_grad()
def calculate_metrics(predictions, labels, mask):
    label_names = ['S', 'B-O', 'B-N', 'I-O', 'I-N']
    metrics = {label: {'correct': 0, 'total': 0, 'predicted': 0} 
              for label in label_names}
    
    valid_preds = predictions[mask].cpu().numpy()
    valid_labels = labels[mask].cpu().numpy()
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, 
        valid_preds, 
        labels=range(len(label_names)), 
        zero_division=0
    )
    
    # Calculate overall accuracy
    accuracy = (valid_preds == valid_labels).mean()
    
    # Prepare detailed metrics
    detailed_metrics = {}
    for i, label in enumerate(label_names):
        detailed_metrics[label] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    # Add macro and weighted averages
    detailed_metrics['macro_avg'] = {
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean()
    }
    
    detailed_metrics['accuracy'] = accuracy
    
    return detailed_metrics

def print_metrics(metrics, phase="Training"):
    print(f"\n{phase} Metrics:")
    print("-" * 50)
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    for label in ['S', 'B-O', 'B-N', 'I-O', 'I-N']:
        print(f"\n{label}:")
        print(f"Precision: {metrics[label]['precision']:.4f}")
        print(f"Recall: {metrics[label]['recall']:.4f}")
        print(f"F1-score: {metrics[label]['f1']:.4f}")
    
    # Print average metrics
    print("\nOverall metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Avg Precision: {metrics['macro_avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {metrics['macro_avg']['recall']:.4f}")
    print(f"Macro Avg F1-score: {metrics['macro_avg']['f1']:.4f}")
    print("-" * 50)

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_f1 = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_masks = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fasttext_embeds = batch['fasttext_embeds'].to(device)
            labels = batch['labels'].to(device)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask, fasttext_embeds)
                    logits_view = outputs.view(-1, outputs.shape[-1])
                    labels_view = labels.view(-1)
                    loss = criterion(logits_view, labels_view)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask, fasttext_embeds)
                logits_view = outputs.view(-1, outputs.shape[-1])
                labels_view = labels.view(-1)
                loss = criterion(logits_view, labels_view)
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=-1)
                mask = (labels != -100)
                
                all_preds.append(predictions[mask])
                all_labels.append(labels[mask])
                all_masks.append(mask)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_preds = torch.cat(all_preds)
        train_labels = torch.cat(all_labels)
        train_metrics = calculate_metrics(train_preds, train_labels, torch.ones_like(train_preds, dtype=torch.bool))
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_labels = []
        all_val_masks = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                fasttext_embeds = batch['fasttext_embeds'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, fasttext_embeds)
                logits_view = outputs.view(-1, outputs.shape[-1])
                labels_view = labels.view(-1)
                loss = criterion(logits_view, labels_view)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=-1)
                mask = (labels != -100)
                
                all_val_preds.append(predictions[mask])
                all_val_labels.append(labels[mask])
                all_val_masks.append(mask)
        
        # Calculate validation metrics
        val_preds = torch.cat(all_val_preds)
        val_labels = torch.cat(all_val_labels)
        val_metrics = calculate_metrics(val_preds, val_labels, torch.ones_like(val_preds, dtype=torch.bool))
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        
        print_metrics(train_metrics, "Training")
        print_metrics(val_metrics, "Validation")
        
        # Save best model
        if val_metrics['macro_avg']['f1'] > best_f1:
            best_f1 = val_metrics['macro_avg']['f1']
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"New best model saved with F1 score: {best_f1:.4f}")

def load_and_process_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    df['processed_labels'] = df['Output Labels'].apply(ast.literal_eval)
    df['full_text'] = df['Context']
    
    texts = df['full_text'].tolist()
    labels = df['processed_labels'].tolist()
    
    valid_samples = [(text, label) for text, label in zip(texts, labels) if label]
    texts, labels = zip(*valid_samples)
    
    return train_test_split(texts, labels, test_size=0.2, random_state=42)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, './data/Dataset_nlp_project_LSTM.csv')
    
    print("Loading and processing data...")
    train_texts, val_texts, train_labels, val_labels = load_and_process_data(data_path)
    
    print("Training FastText model...")
    sentences = [text.split() for text in train_texts]
    ft_model = FastText(sentences, vector_size=300, window=5, min_count=1, workers=1)
    
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Creating datasets...")
    train_dataset = TokenClassificationDataset(train_texts, train_labels, tokenizer, ft_model)
    val_dataset = TokenClassificationDataset(val_texts, val_labels, tokenizer, ft_model)
    
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
    
    print("Initializing model...")
    model = FastTextBertTokenClassifier(num_labels=5).to(device)
    
    print("Starting training...")
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()