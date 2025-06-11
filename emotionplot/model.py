
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model names
FAST_MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"
ACCURATE_MODEL = "SamLowe/roberta-base-go_emotions"

# Load both tokenizers and models
tokenizers = {
    "fast": AutoTokenizer.from_pretrained(FAST_MODEL),
    "accurate": AutoTokenizer.from_pretrained(ACCURATE_MODEL)
}

models = {
    "fast": AutoModelForSequenceClassification.from_pretrained(FAST_MODEL),
    "accurate": AutoModelForSequenceClassification.from_pretrained(ACCURATE_MODEL)
}

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in models.values():
    model.to(device).eval()

# Label maps (identical between the two)
id2label = models["fast"].config.id2label

def predict_emotions(df, text_column="chunk", top_k=3, batch_size=32, model_type="fast"):
    print(f"[predict_emotions] Using model: {model_type}")

    model = models[model_type]
    tokenizer = tokenizers[model_type]

    texts = df[text_column].tolist()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.cpu())

    probs = torch.cat(all_probs, dim=0)

    top_emotions = []
    predicted_labels = []

    for row in probs:
        top_indices = torch.topk(row, top_k).indices.tolist()
        top_scores = [round(row[i].item(), 3) for i in top_indices]
        top_labels = [id2label[i] for i in top_indices]

        top_emotions.append(dict(zip(top_labels, top_scores)))
        predicted_labels.append(top_labels[0])

    df["Predicted_Emotion"] = predicted_labels
    df["Top_3_Emotions"] = top_emotions

    print("[predict_emotions] Prediction complete.")
    return df
