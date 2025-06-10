
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

# Get ID-to-label mapping from model config
id2label = model.config.id2label


def predict_emotion(df):
    # Tokenize input sentence
    inputs = tokenizer(df, return_tensors="pt", truncation=True, padding=True, max_length=512)
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Get the predicted emotion label
    predicted_label = torch.argmax(probabilities).item()
    df["Predicted_Emotion_ID"] = df["cleaned_chunk"].apply(predicted_label)
    df["Predicted_Emotion"] = df["Predicted_Emotion_ID"].map(id2label)
    return df



# Load your DataFrame (assuming it's called df)

# Apply model to DataFrame (assuming df["chunk"] exists)

# Print results
