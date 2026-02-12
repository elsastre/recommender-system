"""Test script to generate recommendations using a trained NCF model.

This script loads a preprocessed dataset and a saved model, then prints
the movies recommended for a specific user.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from src.preprocess import DataProcessor

# 1. Load data and model
processor = DataProcessor('data/u.data')
df = processor.load_and_clean()
model = tf.keras.models.load_model('models/recommender_v1.h5', compile=False)

def get_recommendations(user_id, top_k=5):
    """Generate a list of recommended movie IDs for a user.

    From the original user ID, obtain the internal index, filter out movies
    the user has already seen, predict ratings for the rest and return the
    top_k titles with the highest estimated score.

    Args:
        user_id (int): Original user identifier in the dataset.
        top_k (int, optional): Number of recommendations to generate.

    Returns:
        list: List of original movie IDs for the recommended movies.
    """
    # Get the internal index for the user
    user_idx = df[df['user_id'] == user_id]['user_idx'].iloc[0]

    # Identify movies the user has already seen (do not recommend them)
    watched_movies = df[df['user_id'] == user_id]['movie_idx'].unique()

    # All movies available in the dataset
    all_movie_indices = df['movie_idx'].unique()

    # Candidate movies (those the user has not seen)
    candidate_movies = np.array([m for m in all_movie_indices if m not in watched_movies])

    # Prepare model input
    user_input = np.array([user_idx] * len(candidate_movies))

    # Predict scores
    predictions = model.predict([user_input, candidate_movies], verbose=0).flatten()

    # Get indices of top scores
    top_indices = predictions.argsort()[-top_k:][::-1]
    recommended_indices = candidate_movies[top_indices]

    # Translate indices to original IDs
    return [processor.movie_map[idx] for idx in recommended_indices]

# TEST: Choose a user_id from your dataset (e.g. 1, 10, 50)
user_test = 1
print(f"Recommendations for user {user_test}:")
print(get_recommendations(user_id=user_test))