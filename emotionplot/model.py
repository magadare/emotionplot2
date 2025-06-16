from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import nltk

nltk.download("punkt")

FAST_MODEL_PATH = "models/fast"
ACCURATE_MODEL_PATH = "models/accurate"

# Cache for loaded models/tokenizers
_loaded_models = {}
_loaded_tokenizers = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer(model_type: str):
    if model_type not in ["fast", "accurate"]:
        raise KeyError(f"Invalid model_type: {model_type}. Choose from ['fast', 'accurate'].")

    if model_type not in _loaded_models:
        print(f"[model.py] Loading model/tokenizer: {model_type}")
        model_path = FAST_MODEL_PATH if model_type == "fast" else ACCURATE_MODEL_PATH

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(_device).eval()

        _loaded_tokenizers[model_type] = tokenizer
        _loaded_models[model_type] = model

    return _loaded_models[model_type], _loaded_tokenizers[model_type]


def predict_emotions(df, text_column="chunk", top_k=3, batch_size=32, model_type="accurate"):
    print(f"[predict_emotions] Using model: {model_type}")
    model, tokenizer = get_model_and_tokenizer(model_type)
    id2label = model.config.id2label

    texts = df[text_column].tolist()
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

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

