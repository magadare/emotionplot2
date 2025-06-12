
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk

nltk.download('punkt')

# Load GoEmotions tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

# Get ID-to-label mapping from model config
id2label = model.config.id2label

def predict_emotions(df, text_column="chunk"):
    predicted_ids = []
    predicted_labels = []

    for text in df[text_column]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        predicted_ids.append(pred_id)
        predicted_labels.append(id2label[pred_id])

    df["Predicted_Emotion_ID"] = predicted_ids
    df["Predicted_Emotion"] = predicted_labels
    return df



# Load your DataFrame (assuming it's called df)

# Apply model to DataFrame (assuming df["chunk"] exists)

# Print results
