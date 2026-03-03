"""
Main module for the Recommendation API.

This service exposes a Neural Collaborative Filtering (NCF) model trained on
the MovieLens dataset, allowing retrieval of personalized recommendations
and user history via REST endpoints built with FastAPI.
"""

import os
import time
import logging
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Import our custom TMDB client
from src.tmdb_client import get_movie_poster
from src.preprocess import DataProcessor

# Load environment variables (TMDB API Key)
load_dotenv()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("recommender-api")

# --- DATA MODELS (PYDANTIC) ---

class Recommendation(BaseModel):
    """Represents a single movie recommendation."""
    rank: int
    movie_id: int
    title: str
    confidence_score: float = Field(..., ge=0, le=1)
    poster_url: str  

class PredictResponse(BaseModel):
    """Response structure for the list of recommendations."""
    user_id: int
    is_cold_start: bool = False
    message: str = "Personalized recommendations generated successfully."
    recommendations: List[Recommendation]
    inference_time_ms: float  

class MovieItem(BaseModel):
    """Represents a movie from the user's history."""
    movie_id: int
    title: str
    rating: float
    poster_url: str

class UserHistoryResponse(BaseModel):
    """Response structure for the user's previously highly-rated movies."""
    user_id: int
    history: List[MovieItem]

# --- APP INITIALIZATION ---

app = FastAPI(
    title="Neural Collaborative Filtering API V2",
    description="Inference service with TMDB posters, Cold Start, and Embedding Similarity.",
    version="2.0.0"
)

# --- ASSET LOADING ---

try:
    logger.info("Starting asset loading for the model...")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Load the pre-trained model
    model = tf.keras.models.load_model('models/recommender_v1.keras', compile=False)

    # Initialize the data processor to retrieve mappings (1M dataset)
    processor = DataProcessor('data/ratings.dat') 
    df = processor.load_and_clean()

    # Load and map movie metadata (titles)
    items = pd.read_csv(
        'data/movies.dat', 
        sep='::', 
        engine='python',
        names=['movie_id', 'title', 'genres'], 
        encoding='latin-1'
    )
    movie_titles = dict(zip(items['movie_id'], items['title']))

    # --- COLD START PREPARATION ---
    logger.info("Calculating global top movies for Cold Start...")
    movie_stats = df.groupby('movie_id').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    top_global_df = movie_stats[movie_stats['num_ratings'] > 1000].sort_values(by='avg_rating', ascending=False)
    TOP_GLOBAL_MOVIES = top_global_df.head(50)['movie_id'].tolist()

    # --- EMBEDDINGS EXTRACTION (For Item-to-Item Similarity) ---
    logger.info("Extracting movie embeddings from the neural network...")
    movie_embed_weights = None
    
    # We dynamically search for the movie embedding layer by shape or name
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding) and layer.input_dim == len(processor.movie_map):
            movie_embed_weights = layer.get_weights()[0]
            break
            
    if movie_embed_weights is None:
        for layer in model.layers:
            if 'movie' in layer.name.lower() and 'embed' in layer.name.lower():
                movie_embed_weights = layer.get_weights()[0]
                break

    if movie_embed_weights is not None:
        logger.info(f"Embeddings extracted successfully. Matrix shape: {movie_embed_weights.shape}")
    else:
        logger.warning("Could not automatically extract embeddings. Similarity endpoint may fail.")

    logger.info("Asset loading completed successfully.")

except Exception as e:
    logger.error(f"Critical error during initialization: {e}")
    raise e

# --- ENDPOINTS ---

@app.get("/", tags=["Health Check"])
def health_check():
    """Verify the API status and model version."""
    return {"status": "online", "model_version": "2.0.0"}

@app.get("/user/{user_id}/history", response_model=UserHistoryResponse, tags=["User Data"])
def get_user_history(user_id: int, limit: int = Query(5, description="Number of historical movies to fetch")):
    """Get the top-rated movies a user has already watched."""
    if user_id not in df['user_id'].unique():
        return UserHistoryResponse(user_id=user_id, history=[])
    
    user_data = df[df['user_id'] == user_id].sort_values(by='rating', ascending=False).head(limit)
    
    history_items = []
    for _, row in user_data.iterrows():
        m_id = row['movie_id']
        title = movie_titles.get(m_id, "Unknown Title")
        poster = get_movie_poster(title) 
        
        history_items.append(MovieItem(movie_id=int(m_id), title=title, rating=float(row['rating']), poster_url=poster))
        
    return UserHistoryResponse(user_id=user_id, history=history_items)

@app.get("/recommend/{user_id}", response_model=PredictResponse, tags=["Predictions"])
def get_recommendations(user_id: int, k: int = Query(5, gt=0, le=50)):
    """Generate recommendations enriched with TMDB posters."""
    start_time = time.time()

    try:
        if user_id not in df['user_id'].unique():
            cold_start_recs = []
            for rank, m_id in enumerate(TOP_GLOBAL_MOVIES[:k], 1):
                title = movie_titles.get(m_id, "Unknown Title")
                poster = get_movie_poster(title)
                cold_start_recs.append(Recommendation(rank=rank, movie_id=int(m_id), title=title, confidence_score=1.0, poster_url=poster))
            inference_time = round((time.time() - start_time) * 1000, 2)
            return PredictResponse(user_id=user_id, is_cold_start=True, message="Welcome! Here are some all-time favorites to get you started.", recommendations=cold_start_recs, inference_time_ms=inference_time)

        user_idx = df[df['user_id'] == user_id]['user_idx'].iloc[0]
        watched_movies = df[df['user_id'] == user_id]['movie_idx'].unique()
        all_movie_indices = df['movie_idx'].unique()
        candidate_movies = np.array([m for m in all_movie_indices if m not in watched_movies])

        user_input = np.array([user_idx] * len(candidate_movies))
        predictions = model.predict([user_input, candidate_movies], verbose=0).flatten()

        top_indices = predictions.argsort()[-k:][::-1]
        recommended_movie_indices = candidate_movies[top_indices]

        results = []
        for i, idx in enumerate(recommended_movie_indices):
            m_id = processor.movie_map[idx]
            title = movie_titles.get(m_id, "Unknown Title")
            poster = get_movie_poster(title) 
            results.append(Recommendation(rank=i + 1, movie_id=int(m_id), title=title, confidence_score=float(predictions[top_indices[i]]), poster_url=poster))

        inference_time = round((time.time() - start_time) * 1000, 2)
        return PredictResponse(user_id=user_id, is_cold_start=False, message="Personalized recommendations generated successfully.", recommendations=results, inference_time_ms=inference_time)

    except Exception as e:
        logger.error(f"Internal error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations.")

@app.get("/movie/{movie_id}/similar", response_model=PredictResponse, tags=["Predictions"])
def get_similar_movies(movie_id: int, k: int = Query(5, gt=0, le=20)):
    """Find similar movies using Cosine Similarity on the learned embeddings."""
    if movie_embed_weights is None:
        raise HTTPException(status_code=500, detail="Embeddings were not loaded successfully.")
        
    if movie_id not in items['movie_id'].values:
        raise HTTPException(status_code=404, detail="Movie not found in the original database.")
        
    start_time = time.time()
    
    try:
        # Reverse map to find the neural network's internal index for this movie
        reverse_movie_map = {v: k for k, v in processor.movie_map.items()}
        movie_idx = reverse_movie_map[movie_id]
        
        # Get the specific target vector (1D array converted to 2D for sklearn)
        target_vector = movie_embed_weights[movie_idx].reshape(1, -1)
        
        # Calculate Cosine Similarity against all other movies simultaneously
        similarities = cosine_similarity(target_vector, movie_embed_weights).flatten()
        
        # Get top K indices (we take k+1 because the movie itself will be #1 with a score of 1.0)
        similar_indices = similarities.argsort()[-(k+1):][::-1]
        
        results = []
        rank = 1
        for idx in similar_indices:
            if idx == movie_idx:
                continue # Skip the exact same movie
            
            m_id = processor.movie_map[idx]
            title = movie_titles.get(m_id, "Unknown Title")
            poster = get_movie_poster(title)
            
            results.append(
                Recommendation(
                    rank=rank,
                    movie_id=int(m_id),
                    title=title,
                    confidence_score=float(similarities[idx]), # We use Cosine Similarity as the score
                    poster_url=poster
                )
            )
            rank += 1
            if rank > k:
                break
                
        inference_time = round((time.time() - start_time) * 1000, 2)
        
        return PredictResponse(
            user_id=0, # Not applicable for item-to-item
            is_cold_start=False,
            message=f"Movies similar to '{movie_titles.get(movie_id)}' found.",
            recommendations=results,
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute similarity.")