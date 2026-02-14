import torch
import json
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from .config import MODEL_PATH


model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

model.eval()

with open(f"{MODEL_PATH}/label_map.json", "r") as f:
    id2label = json.load(f)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,padding=True,max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        predicted_class_id = torch.argmax(probs, dim=1).item()

    predicted_label = id2label[str(predicted_class_id)]
    confidence = probs[0][predicted_class_id].item()

    return predicted_label, confidence



# if __name__ == "__main__":
    
#     test_text = "A child playing in a sunny meadow."

#     label, confidence = predict(test_text)

#     print("Text:", test_text)
#     print("Predicted Label:", label)
#     print("Confidence:", round(confidence * 100, 2), "%")
