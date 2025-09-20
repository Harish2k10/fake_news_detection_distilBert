import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def predict(text, model_path="models/best_model"):
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        prediction_id = torch.argmax(outputs.logits, dim=1).item()


    id2label = getattr(model.config, 'id2label', None)
    if isinstance(id2label, dict) and prediction_id in id2label:
        return id2label[prediction_id]

    return "FAKE" if prediction_id == 0 else "REAL"



import sys

if __name__ == "__main__":
    # Case 1: Run with command-line argument
    if len(sys.argv) > 1:
        text = sys.argv[1]
        result = predict(text)
        print(f"Prediction: {result}")

    # Case 2: Interactive mode
    else:
        print("Interactive mode. Type a news sentence (or 'quit' to exit).")
        while True:
            text = input("\nEnter news text: ")
            if text.lower() in ["quit", "exit"]:
                break
            result = predict(text)
            print(f"Prediction: {result}")
