import json
from hashlib import md5
from google.cloud import storage


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
