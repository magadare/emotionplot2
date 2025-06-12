FROM python:3.10.6-slim

COPY minimal_requirements.txt minimal_requirements.txt
RUN pip install --no-cache-dir -r minimal_requirements.txt

COPY api api
COPY emotionplot emotionplot


RUN python -m nltk.downloader punkt punkt_tab

CMD ["sh", "-c", "uvicorn api.api:app --host 0.0.0.0 --port $PORT"]
