from fastapi import FastAPI, Query, HTTPException
from emotionplot.data import get_novel, clean_gutenberg_text
from emotionplot.preprocessing import preprocessing, chunk_by_sentences
from emotionplot.model import predict_emotions
from nltk.tokenize import sent_tokenize
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#http://127.0.0.1:8000/extract/?url=https%3A%2F%2Fwww.gutenberg.org%2Febooks%2F1661
@app.get("/extract/")
def extract_novel(url: str = Query(..., description="Project Gutenberg novel URL")):
    try:
        raw_text = get_novel(url)
        clean_text = clean_gutenberg_text(raw_text)
        return {"status": "success", "text": clean_text[:1000] + "..."}  # Return a preview
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/chunk/")
def chunk_text(
    text: str = Query(..., description="Raw novel text"),
    num_chunks: int = Query(3, ge=1, le=7, description="Number of chunks (1–7)")
):
    try:
        preprocessed = preprocessing(text)
        sentences = sent_tokenize(preprocessed)
        total_sentences = len(sentences)

        # Avoid division by zero
        if total_sentences == 0:
            raise ValueError("The input text contains no sentences.")

        sentences_per_chunk = max(1, total_sentences // num_chunks)

        df = chunk_by_sentences(preprocessed, sentences_per_chunk)

        return {
            "num_chunks": num_chunks,
            "sentences_per_chunk": sentences_per_chunk,
            "chunks": df["chunk"].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#http://127.0.0.1:8000/extract-and-chunk/?url=https%3A%2F%2Fwww.gutenberg.org%2Febooks%2F1661&sentences_per_chunk=3
@app.get("/extract-and-chunk/")
def extract_and_chunk(
    url: str = Query(..., description="Project Gutenberg novel URL"),
    sentences_per_chunk: int = Query(3, ge=1, le=7, description="Number of sentences per chunk (e.g. 3)")
):
    try:
        # Step 1: Fetch and clean text
        raw_text = get_novel(url)
        clean_text = clean_gutenberg_text(raw_text)

        # Step 2: Preprocess and split into sentences
        preprocessed = preprocessing(clean_text)
        sentences = sent_tokenize(preprocessed)
        total_sentences = len(sentences)

        if total_sentences == 0:
            raise ValueError("The input text contains no sentences.")

        # Step 3: Chunk it
        df = chunk_by_sentences(preprocessed, sentences_per_chunk)
        num_chunks = len(df)

        return {
            "status": "success",
            "book_url": url,
            "total_sentences": total_sentences,
            "sentences_per_chunk": sentences_per_chunk,
            "num_chunks": num_chunks,
            "chunks": df["chunk"].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/analyze/")
def full_emotion_pipeline(
    url: str = Query(..., description="Project Gutenberg novel URL"),
    sentences_per_chunk: int = Query(3, ge=1, le=7),
    model: str = Query("fast", enum=["fast", "accurate"], description="Choose 'fast' or 'accurate' model")
):
    try:
        print("Step 1: Getting novel...")
        raw_text = get_novel(url)
        print("Step 2: Preprocessing...")
        clean_text = clean_gutenberg_text(raw_text)
        preprocessed = preprocessing(clean_text)

        print("Step 3: Chunking...")
        sentences = sent_tokenize(preprocessed)
        if not sentences:
            raise ValueError("No sentences found.")
        df_chunks = chunk_by_sentences(preprocessed, sentences_per_chunk)

        print("Step 4: Predicting emotions...")
        df_with_preds = predict_emotions(df_chunks, top_k=3, model_type=model)

        print("Step 5: Returning results.")
        return {
            "status": "success",
            "model_used": model,
            "book_url": url,
            "sentences_per_chunk": sentences_per_chunk,
            "num_chunks": len(df_with_preds),
            "emotions": df_with_preds[["chunk", "Predicted_Emotion", "Top_3_Emotions"]].to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
