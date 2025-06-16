FROM python:3.10.6-slim

RUN apt-get update && apt-get install -y build-essential && apt-get clean


COPY minimal_requirements.txt minimal_requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r minimal_requirements.txt


COPY api/ ./api/
COPY emotionplot/ ./emotionplot/

# Copy local HuggingFace models
COPY models/ ./models/


RUN python -m nltk.downloader punkt punkt_tab

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]
