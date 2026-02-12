"""Construcción de modelo de filtrado colaborativo neuronal (NCF)."""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_ncf_model(num_users: int, num_movies: int, embedding_size: int = 50) -> models.Model:
    """Construye y compila un modelo de Filtrado Colaborativo Neuronal (NCF).

    El modelo utiliza dos ramas de embedding (usuario y película), las concatena
    y pasa el vector resultante a través de una serie de capas densas (MLP) para
    predecir la calificación.

    Args:
        num_users (int): Cantidad total de usuarios únicos en el dataset.
        num_movies (int): Cantidad total de películas únicas en el dataset.
        embedding_size (int, optional): Dimensión del espacio latente para los
            embeddings. Por defecto es 50.

    Returns:
        tensorflow.keras.models.Model: Modelo de Keras compilado con optimizador Adam,
        función de pérdida de error cuadrático medio (MSE) y métrica MAE.
    """
    # Entradas
    user_input = layers.Input(shape=(1,), name='user_input')
    movie_input = layers.Input(shape=(1,), name='movie_input')

    # Embeddings
    user_embedding = layers.Embedding(num_users, embedding_size, name='user_emb')(user_input)
    movie_embedding = layers.Embedding(num_movies, embedding_size, name='movie_emb')(movie_input)

    # Flatten y concatenación
    user_vec = layers.Flatten()(user_embedding)
    movie_vec = layers.Flatten()(movie_embedding)
    concat = layers.Concatenate()([user_vec, movie_vec])

    # Capas densas (MLP)
    dense_1 = layers.Dense(64, activation='relu')(concat)
    dense_2 = layers.Dense(32, activation='relu')(dense_1)
    output = layers.Dense(1, activation='sigmoid')(dense_2)

    model = models.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model