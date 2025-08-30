import os
import pandas as pd

# converts 6 labels into binary
def convert_to_binary(label):
    if label in ["pants-fire", "false", "barely-true"]:
        return 0   # Fake
    else:
        return 1   # Real

# Load and preprocess LIAR dataset
def load_and_preprocess():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    data_dir = os.path.join(BASE_DIR, "data")

    col_names = ["id","label","statement","subject","speaker","job_title","state","party_affiliation",
                 "barely_true_counts","false_counts","half_true_counts","mostly_true_counts","pants_fire_counts","context"]

    # Read TSVs
    train_df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t", header=None, names=col_names)
    val_df   = pd.read_csv(os.path.join(data_dir, "valid.tsv"), sep="\t", header=None, names=col_names)
    test_df  = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t", header=None, names=col_names)

    # Convert labels into binary
    for df in [train_df, val_df, test_df]:
        df["label"] = df["label"].apply(convert_to_binary)
        df.rename(columns={"statement": "text"}, inplace=True)  # keep consistent column name

    # Save CSVs
    train_df[["text","label"]].to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df[["text","label"]].to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df[["text","label"]].to_csv(os.path.join(data_dir, "test.csv"), index=False)

    print("[INFO] Preprocessing done. Saved train.csv, val.csv, test.csv with binary labels.")

if __name__ == "__main__":
    load_and_preprocess()
