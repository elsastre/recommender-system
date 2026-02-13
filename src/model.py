"""
Neural Collaborative Filtering (NCF) Model Architecture.

This module defines the deep learning architecture for the recommendation system,
combining Latent Factor models with Multi-Layer Perceptrons (MLP).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_ncf_model(
    num_users: int, 
    num_movies: int, 
    embedding_size: int = 50
) -> models.Model:
    """
    Build and compile a Neural Collaborative Filtering (NCF) model.

    The architecture consists of two parallel embedding branches (User and Item)
    that are concatenated and passed through a Multi-Layer Perceptron (MLP) 
    to capture non-linear interactions.

    Args:
        num_users (int): Total count of unique users (dimension of User Embedding).
        num_movies (int): Total count of unique items (dimension of Movie Embedding).
        embedding_size (int, optional): Dimension of the latent space. Defaults to 50.

    Returns:
        models.Model: A compiled Keras model ready for training.
    """
    
    # --- Input Layer ---
    # We expect single integers (indices) for both user and movie
    user_input = layers.Input(shape=(1,), name='user_input')
    movie_input = layers.Input(shape=(1,), name='movie_input')

    # --- Embedding Layer ---
    # Maps sparse indices into dense, continuous latent vectors
    user_embedding = layers.Embedding(
        input_dim=num_users, 
        output_dim=embedding_size, 
        name='user_embedding'
    )(user_input)
    
    movie_embedding = layers.Embedding(
        input_dim=num_movies, 
        output_dim=embedding_size, 
        name='movie_embedding'
    )(movie_input)

    # --- Flattening ---
    # Convert (batch, 1, embedding_size) to (batch, embedding_size)
    user_vec = layers.Flatten(name='user_flatten')(user_embedding)
    movie_vec = layers.Flatten(name='movie_flatten')(movie_embedding)

    # --- Interaction Layer (Concatenation) ---
    # Combining both latent vectors to feed the MLP
    concat = layers.Concatenate(name='interaction_layer')([user_vec, movie_vec])

    # --- Multi-Layer Perceptron (MLP) ---
    # Dense layers with ReLU to capture complex patterns
    dense_1 = layers.Dense(64, activation='relu', name='fully_connected_1')(concat)
    dense_2 = layers.Dense(32, activation='relu', name='fully_connected_2')(dense_1)

    # --- Output Layer ---
    # Sigmoid activation to output a probability/score between 0 and 1
    output = layers.Dense(1, activation='sigmoid', name='prediction')(dense_2)

    # --- Model Compilation ---
    model = models.Model(inputs=[user_input, movie_input], outputs=output)
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    return model