import json
from hashlib import md5
from google.cloud import storage
from collections import defaultdict


BUCKET_NAME = "emotionplot-results"

def generate_novel_id(url: str) -> str:
    """Generates a unique ID for a novel based on its URL.
    This ID is used to store and retrieve the novel's data in Google Cloud Storage.
    Args:
        url (str): The URL of the novel.
    Returns:
        str: A unique ID for the novel.
    """
    return md5(url.encode()).hexdigest()

def generate_poem_id(text: str) -> str:
    """Generates a unique ID for user-submitted text (e.g. poems)."""
    return md5(text.encode("utf-8")).hexdigest()[:16]

def upload_to_gcs(data: dict, bucket_name: str, blob_name: str):
    """Uploads a dictionary to Google Cloud Storage as a JSON file.
    Args:
        data (dict): The data to upload.
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The name of the blob (file) in the bucket.
    """
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dictionary")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

def download_from_gcs_if_exists(bucket_name: str, blob_name: str):
    """Downloads a JSON file from Google Cloud Storage if it exists.
    Args:
        bucket_name (str): The name of the GCS bucket.
        blob_name (str): The name of the blob (file) in the bucket.
    Returns:
        dict or None: The content of the JSON file as a dictionary, or None if the file does not exist.
    """
    if not isinstance(bucket_name, str) or not isinstance(blob_name, str):
        raise TypeError("`bucket_name` and `blob_name` must be strings")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if blob.exists():
        content = blob.download_as_text()
        return json.loads(content)
    return None



def compute_emotion_profile(emotions_data):
    """
    Compute emotion profile from emotions data.
    Handles both old format (list) and new format (dict) for Top_3_Emotions.
    """
    emotion_counts = defaultdict(float)
    total_chunks = len(emotions_data)

    for emotion_entry in emotions_data:
        top_3 = emotion_entry.get("Top_3_Emotions", {})

        # Handle both old format (list) and new format (dict)
        if isinstance(top_3, list):
            # Old format: convert list to dict with equal weights
            # Assume the list contains emotion names in order of preference
            for i, emotion in enumerate(top_3[:3]):  # Take max 3 emotions
                if isinstance(emotion, str):
                    # Give higher weight to first emotions in the list
                    weight = 1.0 / (i + 1)  # 1.0, 0.5, 0.33
                    emotion_counts[emotion] += weight
                elif isinstance(emotion, dict) and 'emotion' in emotion:
                    # Handle case where list contains emotion dicts
                    weight = emotion.get('score', 1.0 / (i + 1))
                    emotion_counts[emotion['emotion']] += weight
        elif isinstance(top_3, dict):
            # New format: dictionary with emotion -> score mapping
            for emotion, score in top_3.items():
                if isinstance(score, (int, float)):
                    emotion_counts[emotion] += score
                else:
                    # Fallback: treat as binary presence
                    emotion_counts[emotion] += 1.0
        else:
            # Fallback: use primary emotion if available
            primary_emotion = emotion_entry.get("Predicted_Emotion")
            if primary_emotion:
                emotion_counts[primary_emotion] += 1.0

    # Normalize by total chunks
    if total_chunks > 0:
        emotion_profile = {emotion: count/total_chunks for emotion, count in emotion_counts.items()}
    else:
        emotion_profile = {}

    return emotion_profile




def load_all_profiles(bucket_name, prefix="emotion_results/books/"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    emotion_profiles = {}
    url_lookup = {}

    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".json"):
            content = blob.download_as_string()
            data = json.loads(content)

            profile = compute_emotion_profile(data["emotions"])
            emotion_profiles[blob.name] = profile

            # ✅ Save the book URL for later use
            url_lookup[blob.name] = data.get("book_url", "Unknown")

    return emotion_profiles, url_lookup
