"""Script de prueba para generar recomendaciones usando un modelo NCF entrenado.

Este script carga un dataset preprocesado y un modelo guardado, y muestra
las películas recomendadas para un usuario específico.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from src.preprocess import DataProcessor

# 1. Cargar datos y modelo
processor = DataProcessor('data/u.data')
df = processor.load_and_clean()
model = tf.keras.models.load_model('models/recommender_v1.h5', compile=False)

def get_recommendations(user_id, top_k=5):
    """Genera una lista de IDs de películas recomendadas para un usuario.

    A partir del ID original del usuario, obtiene su índice interno, filtra
    las películas que ya ha visto, predice las calificaciones para el resto
    y retorna los `top_k` títulos con mayor puntaje estimado.

    Args:
        user_id (int): Identificador original del usuario en el dataset.
        top_k (int, optional): Cantidad de recomendaciones a generar.

    Returns:
        list: Lista de IDs originales de las películas recomendadas.
    """
    # Obtener el índice del usuario
    user_idx = df[df['user_id'] == user_id]['user_idx'].iloc[0]

    # Identificar qué películas YA vio (para no recomendarlas)
    watched_movies = df[df['user_id'] == user_id]['movie_idx'].unique()

    # Todas las películas disponibles en el dataset
    all_movie_indices = df['movie_idx'].unique()

    # Películas candidatas (las que no ha visto)
    candidate_movies = np.array([m for m in all_movie_indices if m not in watched_movies])

    # Preparar la entrada para el modelo
    user_input = np.array([user_idx] * len(candidate_movies))

    # Predicción de scores
    predictions = model.predict([user_input, candidate_movies], verbose=0).flatten()

    # Obtener los índices de los mejores scores
    top_indices = predictions.argsort()[-top_k:][::-1]
    recommended_indices = candidate_movies[top_indices]

    # Traducir índices a IDs reales
    return [processor.movie_map[idx] for idx in recommended_indices]

# PRUEBA: Elige un user_id de tu dataset (ej: 1, 10, 50)
user_test = 1
print(f"Recomendaciones para el usuario {user_test}:")
print(get_recommendations(user_id=user_test))