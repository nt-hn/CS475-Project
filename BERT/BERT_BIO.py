import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForTokenClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Load data
data_file = "./data/Dataset_nlp_project_BIO.csv"
df = pd.read_csv(data_file)

# Check if context and output labels match in length
for i, row in df.iterrows():
    context = row["Context"]
    output_labels = eval(row["Output Labels"])  # Convert to list
    num_words = len(context.split())
    if len(output_labels) != num_words:
        print(f"Mismatch in row {i}:")
        print(f"Context: {context}")
        print(f"Output Labels: {output_labels}")

# BIO label mapping
label_mapping = {"B-O": 0, "I-O": 1, "B-N": 2, "I-N": 3, "S": 4}
num_labels = len(label_mapping)

# Dataset class
class MisspelledWordDataset(Dataset):
    def __init__(self, contexts, labels, label_mapping, max_len=128):
        self.contexts = contexts
        self.labels = labels
        self.label_mapping = label_mapping
        self.max_len = max_len

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        output_labels = eval(self.labels[idx])  # Convert to list

        # Tokenize text by space
        tokenized_text = context.split()
        aligned_labels = []
        for i, token in enumerate(tokenized_text):
            if i < len(output_labels):
                aligned_labels.append(self.label_mapping[output_labels[i]])
            else:
                aligned_labels.append(self.label_mapping["S"])  # Default label

        # Padding to max_len
        input_ids = [i + 1 for i in range(len(tokenized_text))]
        attention_mask = [1] * len(tokenized_text)
        input_ids += [0] * (self.max_len - len(input_ids))
        attention_mask += [0] * (self.max_len - len(attention_mask))
        aligned_labels += [-100] * (self.max_len - len(aligned_labels))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Context"], df["Output Labels"], test_size=0.2, random_state=42
)
train_texts = train_texts.reset_index(drop=True).tolist()
test_texts = test_texts.reset_index(drop=True).tolist()
train_labels = train_labels.reset_index(drop=True).tolist()
test_labels = test_labels.reset_index(drop=True).tolist()

# Initialize datasets
train_dataset = MisspelledWordDataset(train_texts, train_labels, label_mapping=label_mapping)
test_dataset = MisspelledWordDataset(test_texts, test_labels, label_mapping=label_mapping)

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Load pretrained BERT model
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
model.to(device)

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

# Evaluation function with corrected logic for S labels
def evaluate(model, dataloader, label_mapping):
    model.eval()
    binary_predictions, binary_true_labels = [], []
    s_predictions, s_true_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()

            # Remove padding (-100)
            for pred, label in zip(preds, labels):
                pred_clean = [p for p, l in zip(pred, label) if l != -100]
                label_clean = [l for l in label if l != -100]

                # Check for length mismatch
                if len(pred_clean) != len(label_clean):
                    print(f"Length mismatch: pred={len(pred_clean)}, label={len(label_clean)}")
                    continue

                # Map predictions and labels for binary classification (N vs O)
                binary_pred = [
                    0 if p in [label_mapping["B-O"], label_mapping["I-O"]] else (
                        1 if p in [label_mapping["B-N"], label_mapping["I-N"]] else -1
                    )
                    for p in pred_clean
                ]
                binary_true = [
                    0 if l in [label_mapping["B-O"], label_mapping["I-O"]] else (
                        1 if l in [label_mapping["B-N"], label_mapping["I-N"]] else -1
                    )
                    for l in label_clean
                ]

                # Ensure consistent filtering for -1 (S labels)
                filtered_pred = [p for p, t in zip(binary_pred, binary_true) if t != -1]
                filtered_true = [t for t in binary_true if t != -1]

                binary_predictions.extend(filtered_pred)
                binary_true_labels.extend(filtered_true)

                # Separate predictions and labels for S metrics
                s_pred = [p for p, l in zip(pred_clean, label_clean) if l == label_mapping["S"]]
                s_label = [l for l in label_clean if l == label_mapping["S"]]
                s_predictions.extend(s_pred)
                s_true_labels.extend(s_label)
    
    # Check for empty binary predictions/true labels
    if len(binary_predictions) == 0 or len(binary_true_labels) == 0:
        print("Error: Binary predictions or true labels are empty after filtering.")
        return

    # Binary classification report (N vs O only)
    print("Binary Classification Report (N vs O):")
    print(classification_report(binary_true_labels, binary_predictions, labels=[0, 1], target_names=["O", "N"]))

    # Metrics for S label
    if len(s_predictions) > 0 and len(s_true_labels) > 0:
        print("Metrics for S label:")
        print(classification_report(s_true_labels, s_predictions, labels=[label_mapping["S"]], target_names=["S"]))
    else:
        print("No valid samples for S label metrics.")

# Evaluate on the test set
print("Evaluating on test set...")
evaluate(model, test_dataloader, label_mapping)
