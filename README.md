# Fake News Detection with DistilBERT


## Overview

This project fine-tunes **DistilBERT** to classify news as REAL or FAKE. It uses the **LIAR dataset** for training, applies preprocessing, and includes early stopping to prevent overfitting. The best model is automatically saved for inference.

## Project Structure

* `main.py` 
* `src/preprocess.py`
* `src/train.py` 
* `src/predict.py` 
* `src/evaluate.py` 
* `data/` 
* `models/` 

## Requirements

* Python 3.9+
* PyTorch 
* Transformers, scikit-learn, pandas, tqdm

## Installation

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```


## Data

This project uses the **LIAR dataset** ([link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)).

Steps:

1. Download the dataset (TSV format).
2. Place the dataset files (train.tsv, val.tsv, test.tsv) inside the `data/` directory.
3. Run `main.py`. Preprocessing will automatically convert the TSVs into CSVs.


## Training

Run:

```bash
python main.py
```

* Trains for a few epochs with validation after each epoch
* Saves the best model and tokenizer to `models/best_model`
* Will show metrics per epoch (loss, accuracy, F1)
* "✅ Best model saved!" when validation F1 improves

## Prediction


```bash
python src/predict.py "Your news text here"
```

Output: `REAL` or `FAKE`

Or run interactively:

```bash
python src/predict.py
```

## Speed Tips

* Reduce sequence length: set `max_len=128` in `preprocess.py` for faster training
* Use fewer epochs for quick checks (e.g., `epochs=2`)

## Notes & Assumptions

* Labels: `1 = REAL`, `0 = FAKE`
* Pretrained model: `distilbert-base-uncased`
* Models and tokenizer are saved under `models/best_model`

## License

MIT License
