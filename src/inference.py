"""Servicio de recomendación basado en modelo de filtrado colaborativo neural.

Este módulo proporciona la clase RecommenderService que carga un modelo entrenado
de TensorFlow y un dataset de calificaciones, y genera recomendaciones personalizadas
para usuarios específicos.
"""

import pandas as pd
import numpy as np
import tensorflow as tf

class RecommenderService:
    """Servicio para generar recomendaciones de películas usando un modelo NCF.

    Carga un modelo previamente entrenado y el dataset de calificaciones, construye
    los mapeos necesarios entre identificadores originales e índices internos,
    y expone un método para obtener las mejores recomendaciones para un usuario.
    """

    def __init__(self, model_path, dataset_path):
        """Inicializa el servicio cargando el modelo y el dataset.

        Args:
            model_path (str): Ruta del archivo del modelo Keras entrenado.
            dataset_path (str): Ruta del archivo CSV con las calificaciones.
        """
        self.model = tf.keras.models.load_model(model_path)
        # Necesitamos los datos para saber qué películas ya vio el usuario
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        self.df = pd.read_csv(dataset_path, sep='\t', names=columns)
        
        # Re-creamos los mapeos (en un entorno real, estos se guardan en un JSON)
        self.user_ids = self.df['user_id'].unique().tolist()
        self.movie_ids = self.df['item_id'].unique().tolist()
        self.user2idx = {x: i for i, x in enumerate(self.user_ids)}
        self.movie2idx = {x: i for i, x in enumerate(self.movie_ids)}
        self.idx2movie = {i: x for i, x in enumerate(self.movie_ids)}

    def get_recommendations(self, user_id, top_k=5):
        """Genera una lista de IDs de películas recomendadas para un usuario.

        Filtra las películas que el usuario ya ha visto, predice las calificaciones
        para el resto utilizando el modelo y retorna las `top_k` con mayor puntaje.

        Args:
            user_id (int): Identificador del usuario.
            top_k (int): Cantidad de recomendaciones a generar.

        Returns:
            list: Lista de IDs de películas recomendadas, o un mensaje de error
                  si el usuario no existe en el dataset.
        """
        user_idx = self.user2idx.get(user_id)
        if user_idx is None:
            return "Usuario no encontrado"

        # Películas que el usuario YA vio (no queremos recomendarlas de nuevo)
        movies_watched = self.df[self.df['user_id'] == user_id]['item_id'].values
        
        # Películas que NO ha visto
        movies_not_watched = [m for m in self.movie_ids if m not in movies_watched]
        movies_not_watched_idx = [self.movie2idx.get(m) for m in movies_not_watched]
        
        # Preparamos los vectores para la predicción masiva
        user_input = np.array([user_idx] * len(movies_not_watched_idx))
        movie_input = np.array(movies_not_watched_idx)
        
        # Predicción: ¿Qué puntaje le daría el modelo a cada película no vista?
        ratings = self.model.predict([user_input, movie_input]).flatten()
        
        # Ordenamos y tomamos las mejores K
        top_indices = ratings.argsort()[-top_k:][::-1]
        recommended_movie_ids = [self.idx2movie.get(movies_not_watched_idx[i]) for i in top_indices]
        
        return recommended_movie_ids