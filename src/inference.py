"""Recommendation service based on a Neural Collaborative Filtering model.

This module provides the `RecommenderService` class which loads a trained
TensorFlow model and a ratings dataset, and generates personalized
recommendations for specific users.
"""

import pandas as pd
import numpy as np
import tensorflow as tf

class RecommenderService:
    """Service to generate movie recommendations using an NCF model.

    Loads a pre-trained model and the ratings dataset, builds the required
    mappings between original IDs and internal indices, and exposes a method
    to fetch top recommendations for a user.
    """

    def __init__(self, model_path, dataset_path):
        """Initialize the service by loading the model and dataset.

        Args:
            model_path (str): Path to the trained Keras model file.
            dataset_path (str): Path to the CSV file with ratings.
        """
        self.model = tf.keras.models.load_model(model_path)
        # We need the data to know which movies the user has already seen
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        self.df = pd.read_csv(dataset_path, sep='\t', names=columns)
        
        # Recreate mappings (in a real environment these would be persisted to JSON)
        self.user_ids = self.df['user_id'].unique().tolist()
        self.movie_ids = self.df['item_id'].unique().tolist()
        self.user2idx = {x: i for i, x in enumerate(self.user_ids)}
        self.movie2idx = {x: i for i, x in enumerate(self.movie_ids)}
        self.idx2movie = {i: x for i, x in enumerate(self.movie_ids)}

    def get_recommendations(self, user_id, top_k=5):
        """Generate a list of recommended movie IDs for a user.

        Filters out movies the user has already seen, predicts ratings for the
        remaining movies using the model, and returns the top_k movies by score.

        Args:
            user_id (int): User identifier.
            top_k (int): Number of recommendations to generate.

        Returns:
            list: List of recommended movie IDs, or an error message if the user
                  does not exist in the dataset.
        """
        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            return "User not found"

        # Movies the user HAS already seen (we should not recommend them again)
        movies_watched = self.df[self.df['user_id'] == user_id]['item_id'].values
        
        # Movies the user has NOT seen
        movies_not_watched = [m for m in self.movie_ids if m not in movies_watched]
        movies_not_watched_idx = [self.movie2idx.get(m) for m in movies_not_watched]
        
        # Prepare arrays for batch prediction
        user_input = np.array([user_idx] * len(movies_not_watched_idx))
        movie_input = np.array(movies_not_watched_idx)
        
        # Prediction: what score would the model give to each unseen movie?
        ratings = self.model.predict([user_input, movie_input]).flatten()
        
        # Sort and take the top K
        top_indices = ratings.argsort()[-top_k:][::-1]
        recommended_movie_ids = [self.idx2movie.get(movies_not_watched_idx[i]) for i in top_indices]
        
        return recommended_movie_ids