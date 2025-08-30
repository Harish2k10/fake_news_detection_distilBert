import os
import pandas as pd
from transformers import DistilBertTokenizer
from src.train import NewsDataset, train_model
from src.preprocess import load_and_preprocess

def main():
    if not (os.path.exists("data/train.csv") and os.path.exists("data/val.csv") and os.path.exists("data/test.csv")):
        print("[INFO] Preprocessed CSVs not found. Running preprocess...")
        load_and_preprocess()

    train_df = pd.read_csv("data/train.csv")
    val_df   = pd.read_csv("data/val.csv")

    print(f"[INFO] Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # Tokenizer 
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(texts, labels):
        encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=256
        )
        return encodings, list(labels)

    # now use "text" column from preprocess.py
    train_encodings, train_labels = tokenize(train_df["text"], train_df["label"])
    val_encodings, val_labels     = tokenize(val_df["text"], val_df["label"])

    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset   = NewsDataset(val_encodings, val_labels)

    # Training Model
    model = train_model(
        train_dataset,
        val_dataset,
        tokenizer,
        epochs=5,
        batch_size=32,
        lr=3e-5
    )

if __name__ == "__main__":
    main()
