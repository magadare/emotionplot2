from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from emotionplot.gcs_utils import load_all_profiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class EmotionEntry(BaseModel):
    chunk: str
    Predicted_Emotion: str
    Top_3_Emotions: Dict[str, float]

class RecommendationRequest(BaseModel):
    emotions: List[EmotionEntry]
    top_k: int = Field(
        10,
        ge=1,
        le=20,
        description="Number of similar books to return (1-20)"
        )

all_profiles = load_all_profiles("emotionplot-results")


def recommend_similar_books(new_profile, all_profiles, top_k=20):
    df = pd.DataFrame(all_profiles).fillna(0).T
    new_vector = pd.Series(new_profile).reindex(df.columns).fillna(0).values.reshape(1, -1)
    similarity_scores = cosine_similarity(df.values, new_vector).flatten()

    recommendations = pd.DataFrame({
        "book": df.index,
        "similarity": similarity_scores
    }).sort_values(by="similarity", ascending=False)

    return recommendations.head(top_k)
