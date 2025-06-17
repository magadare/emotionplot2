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


def predict_emotions(df, text_column="chunk", top_k=28, batch_size=32, model_type="accurate"):
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

def predict_emotions_poems(df, text_column="chunk", top_k=28, batch_size=32, model_type="accurate"):
    """
    Predict emotions for poem lines individually.
    Args:
        df (pd.DataFrame): DataFrame containing poem lines
        text_column (str): Column name containing the text to analyze
        top_k (int): Number of top emotions to return
        batch_size (int): Batch size for processing
        model_type (str): Model type to use
    Returns:
        pd.DataFrame: DataFrame with emotion predictions for each line
    """
    print(f"[predict_emotions_poems] Processing {len(df)} individual poem lines")
    print(f"[predict_emotions_poems] Using model: {model_type}")

    model, tokenizer = get_model_and_tokenizer(model_type)
    id2label = model.config.id2label

    texts = df[text_column].tolist()
    print(f"[predict_emotions_poems] Sample texts to analyze:")
    for i, text in enumerate(texts[:3]):
        print(f"  Line {i+1}: '{text}'")

    all_probs = []

    # Process in batches but keep individual predictions
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"[predict_emotions_poems] Processing batch {i//batch_size + 1}, lines {i+1} to {min(i+batch_size, len(texts))}")

        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.cpu())

    # Concatenate all probabilities
    probs = torch.cat(all_probs, dim=0)
    print(f"[predict_emotions_poems] Generated predictions for {len(probs)} lines")

    # Extract top emotions for each line individually
    top_emotions = []
    predicted_labels = []

    for idx, row in enumerate(probs):
        top_indices = torch.topk(row, top_k).indices.tolist()
        top_scores = [round(row[i].item(), 3) for i in top_indices]
        top_labels = [id2label[i] for i in top_indices]

        # Create dictionary format to match original function (for compatibility)
        top_emotions_dict = dict(zip(top_labels, top_scores))

        top_emotions.append(top_emotions_dict)
        predicted_labels.append(top_labels[0])

        # Debug output for first few predictions
        if idx < 3:
            print(f"  Line {idx+1} prediction: {top_labels[0]} (confidence: {top_scores[0]})")

    # Add predictions to dataframe
    df = df.copy()  # Avoid modifying original dataframe
    df["Predicted_Emotion"] = predicted_labels
    df["Top_3_Emotions"] = top_emotions

    print(f"[predict_emotions_poems] Completed predictions for {len(df)} lines")
    return df
    """
    Create individual chunks for each line of the poem.
    Args:
        lines (list): List of preprocessed poem lines
    Returns:
        pd.DataFrame: DataFrame with each line as a separate chunk
    """
    chunks = []

    for i, line in enumerate(lines):
        if line.strip():  # Only process non-empty lines
            chunks.append({
                'chunk': line,  # Keep 'chunk' column name for compatibility with predict_emotions
                'line_number': i + 1,
                'line_text': line
            })

    df = pd.DataFrame(chunks)
    print(f"DEBUG: Created {len(df)} individual chunks")
    print("DEBUG: First few chunks:")
    for idx, row in df.head(3).iterrows():
        print(f"  Chunk {idx}: '{row['chunk']}'")

    return df
