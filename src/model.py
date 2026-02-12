"""Neural Collaborative Filtering (NCF) model construction."""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_ncf_model(num_users: int, num_movies: int, embedding_size: int = 50) -> models.Model:
    """Build and compile a Neural Collaborative Filtering (NCF) model.

    The model uses two embedding branches (user and movie), concatenates them,
    and passes the resulting vector through a small MLP to predict the rating.

    Args:
        num_users (int): Total number of unique users in the dataset.
        num_movies (int): Total number of unique movies in the dataset.
        embedding_size (int, optional): Dimension of the latent embedding space.
            Defaults to 50.

    Returns:
        tensorflow.keras.models.Model: A compiled Keras model using Adam optimizer,
        mean squared error (MSE) loss and MAE metric.
    """
    # Inputs
    user_input = layers.Input(shape=(1,), name='user_input')
    movie_input = layers.Input(shape=(1,), name='movie_input')

    # Embeddings
    user_embedding = layers.Embedding(num_users, embedding_size, name='user_emb')(user_input)
    movie_embedding = layers.Embedding(num_movies, embedding_size, name='movie_emb')(movie_input)

    # Flatten and concatenate
    user_vec = layers.Flatten()(user_embedding)
    movie_vec = layers.Flatten()(movie_embedding)
    concat = layers.Concatenate()([user_vec, movie_vec])

    # Dense layers (MLP)
    dense_1 = layers.Dense(64, activation='relu')(concat)
    dense_2 = layers.Dense(32, activation='relu')(dense_1)
    output = layers.Dense(1, activation='sigmoid')(dense_2)

    model = models.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model