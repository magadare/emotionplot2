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



def compute_emotion_profile(emotions_list):
    emotion_totals = defaultdict(float)
    total_chunks = len(emotions_list)

    for entry in emotions_list:
        top_3 = entry.get("Top_3_Emotions", {})
        for emotion, score in top_3.items():
            emotion_totals[emotion] += score

    for emotion in emotion_totals:
        emotion_totals[emotion] /= total_chunks

    return dict(emotion_totals)




def load_all_profiles(bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    emotion_profiles = {}
    url_lookup = {}

    for blob in bucket.list_blobs():
        if blob.name.endswith(".json"):
            content = blob.download_as_string()
            data = json.loads(content)

            profile = compute_emotion_profile(data["emotions"])
            emotion_profiles[blob.name] = profile

            # ✅ Save the book URL for later use
            url_lookup[blob.name] = data.get("book_url", "Unknown")

    return emotion_profiles, url_lookup
