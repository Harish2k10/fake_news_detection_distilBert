import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

# Dataset Class
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Training Function 
def train_model(train_dataset, val_dataset, tokenizer, epochs=5, batch_size=32, lr=3e-5, patience=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.config.id2label = {0: "FAKE", 1: "REAL"}
    model.config.label2id = {"FAKE": 0, "REAL": 1}
    model.to(device)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer + Loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation 
        model.eval()
        val_loss = 0
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs.logits, dim=-1)
                preds.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average="weighted")

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {acc:.4f} | "
              f"Val F1: {f1:.4f}")

        # Early Stopping 
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            os.makedirs("models/best_model", exist_ok=True)
            model.save_pretrained("models/best_model")
            tokenizer.save_pretrained("models/best_model")
            print("✅ Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered.")
                break

    return model

