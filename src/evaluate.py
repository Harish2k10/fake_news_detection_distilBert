import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from src.train import NewsDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model_path="models/best_model", test_path="data/test.csv"):
    # Load preprocessed test dataset
    df = pd.read_csv(test_path)

    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    encodings = tokenizer(
        list(df['text']),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    dataset = NewsDataset(encodings, list(df['label']))
    dataloader = DataLoader(dataset, batch_size=32)

    # Evaluation loop
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Accuracy on test set: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    evaluate()
