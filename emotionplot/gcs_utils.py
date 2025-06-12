import json
from hashlib import md5
from google.cloud import storage

BUCKET_NAME = "emotionplot-results"

def generate_novel_id(url: str) -> str:
    return md5(url.encode()).hexdigest()

def upload_to_gcs(data: dict, bucket_name: str, blob_name: str):
    client = storage.Client(project="static-hangout-457110-u5")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(data), content_type="application/json")

def download_from_gcs_if_exists(bucket_name: str, blob_name: str):
    client = storage.Client(project="static-hangout-457110-u5")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name) 
    if blob.exists():
        content = blob.download_as_text()
        return json.loads(content)
    return None
