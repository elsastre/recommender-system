"""
Main module for the Recommendation API.

This service exposes a Neural Collaborative Filtering (NCF) model trained on
the MovieLens 100k dataset, allowing retrieval of personalized recommendations
via REST endpoints built with FastAPI.
"""

import logging
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("recommender-api")

# --- DATA MODELS (PYDANTIC) ---

class Recommendation(BaseModel):
    """
    Represents a single movie recommendation.

    Attributes:
        rank (int): Position in the recommendation ranking.
        movie_id (int): Original movie identifier.
        title (str): Movie title.
        confidence_score (float): Probability or score predicted by the model,
            normalized to the range [0, 1].
    """
    rank: int
    movie_id: int
    title: str
    confidence_score: float = Field(..., ge=0, le=1)

class PredictResponse(BaseModel):
    """
    Response structure for the list of recommendations.

    Attributes:
        user_id (int): Identifier of the user for whom recommendations are generated.
        recommendations (List[Recommendation]): Ordered list of recommendations.
    """
    user_id: int
    recommendations: List[Recommendation]

# --- APP INITIALIZATION ---

app = FastAPI(
    title="Neural Collaborative Filtering API",
    description="Inference service for the NCF recommender system.",
    version="1.1.0"
)

# --- ASSET LOADING ---

try:
    logger.info("Starting asset loading for the model...")

    # Load the pre-trained model (.h5) in inference mode
    model = tf.keras.models.load_model('models/recommender_v1.keras', compile=False)

    # Initialize the data processor to retrieve mappings
    from src.preprocess import DataProcessor
    processor = DataProcessor('data/u.data')
    df = processor.load_and_clean()

    # Load and map movie metadata (titles)
    cols = ['movie_id', 'title'] + [f'extra_{i}' for i in range(22)]
    items = pd.read_csv('data/u.item', sep='|', names=cols, encoding='latin-1')
    movie_titles = dict(zip(items['movie_id'], items['title']))

    logger.info("Asset loading completed successfully.")
except Exception as e:
    logger.error(f"Critical error during initialization: {e}")
    raise e

# --- ENDPOINTS ---

@app.get("/", tags=["Health Check"])
def health_check():
    """
    Health check for the API and model version.

    Returns:
        dict: Dictionary with 'online' status and model version.
    """
    return {"status": "online", "model_version": "1.1.0"}

@app.get(
    "/recommend/{user_id}",
    response_model=PredictResponse,
    tags=["Predictions"]
)
def get_recommendations(
    user_id: int,
    k: int = Query(5, gt=0, le=50, description="Number of recommendations to generate")
):
    """
    Generate a list of K recommendations for a user based on their history.

    The process follows three stages:
    1. Filter out movies already seen by the user.
    2. Run batch inference over the remaining catalog using the NCF model.
    3. Rank and retrieve metadata for the Top-K results.

    Args:
        user_id (int): Original user identifier.
        k (int): Number of requested recommendations. Must be between 1 and 50.

    Returns:
        PredictResponse: Object containing the user ID and the recommendation list.

    Raises:
        HTTPException: 404 if the user does not exist, 500 if an internal prediction
            error occurs.
    """
    logger.info(f"Processing recommendation request for User: {user_id} (K={k})")

    # Validate that the user exists in the training space
    if user_id not in df['user_id'].unique():
        logger.warning(f"User ID {user_id} not present in the original data.")
        raise HTTPException(status_code=404, detail="User not found.")

    try:
        # Retrieve internal indices and unseen movies
        user_idx = df[df['user_id'] == user_id]['user_idx'].iloc[0]
        watched_movies = df[df['user_id'] == user_id]['movie_idx'].unique()
        all_movie_indices = df['movie_idx'].unique()
        candidate_movies = np.array([m for m in all_movie_indices if m not in watched_movies])

        # Prepare inputs and run inference
        user_input = np.array([user_idx] * len(candidate_movies))
        predictions = model.predict([user_input, candidate_movies], verbose=0).flatten()

        # Select top results
        top_indices = predictions.argsort()[-k:][::-1]
        recommended_movie_indices = candidate_movies[top_indices]

        results = []
        for i, idx in enumerate(recommended_movie_indices):
            m_id = processor.movie_map[idx]
            results.append(
                Recommendation(
                    rank=i + 1,
                    movie_id=int(m_id),
                    title=movie_titles.get(m_id, "Unknown Title"),
                    confidence_score=float(predictions[top_indices[i]])
                )
            )

        logger.info(f"Pipeline finished. {len(results)} recommendations generated.")
        return PredictResponse(user_id=user_id, recommendations=results)

    except Exception as e:
        logger.error(f"Internal error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations.")