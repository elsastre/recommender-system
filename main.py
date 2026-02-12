"""
Módulo principal de la API de Recomendación.

Este servicio expone un modelo de Filtrado Colaborativo Neuronal (NCF) entrenado
con el dataset MovieLens 100k, permitiendo obtener recomendaciones personalizadas
a través de endpoints REST construidos con FastAPI.
"""

import logging
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("recommender-api")

# --- MODELOS DE DATOS (PYDANTIC) ---

class Recommendation(BaseModel):
    """
    Representa una única recomendación de película.

    Attributes:
        rank (int): Posición en el ranking de recomendaciones.
        movie_id (int): Identificador original de la película.
        title (str): Nombre de la película.
        confidence_score (float): Probabilidad o puntaje predicho por el modelo,
            normalizado en el rango [0, 1].
    """
    rank: int
    movie_id: int
    title: str
    confidence_score: float = Field(..., ge=0, le=1)

class PredictResponse(BaseModel):
    """
    Estructura de respuesta para el listado de recomendaciones.

    Attributes:
        user_id (int): Identificador del usuario para quien se generan las recomendaciones.
        recommendations (List[Recommendation]): Lista ordenada de recomendaciones.
    """
    user_id: int
    recommendations: List[Recommendation]

# --- INICIALIZACIÓN DE LA APP ---

app = FastAPI(
    title="Neural Collaborative Filtering API",
    description="Servicio de inferencia para el sistema de recomendación NCF.",
    version="1.1.0"
)

# --- CARGA DE ACTIVOS ---

try:
    logger.info("Iniciando carga de activos del modelo...")

    # Carga del modelo pre-entrenado (.h5) en modo inferencia
    model = tf.keras.models.load_model('models/recommender_v1.h5', compile=False)

    # Inicialización del procesador de datos para recuperación de mapeos
    from src.preprocess import DataProcessor
    processor = DataProcessor('data/u.data')
    df = processor.load_and_clean()

    # Carga y mapeo de metadatos de películas (Títulos)
    cols = ['movie_id', 'title'] + [f'extra_{i}' for i in range(22)]
    items = pd.read_csv('data/u.item', sep='|', names=cols, encoding='latin-1')
    movie_titles = dict(zip(items['movie_id'], items['title']))

    logger.info("Carga de activos completada exitosamente.")
except Exception as e:
    logger.error(f"Error crítico en la inicialización: {e}")
    raise e

# --- ENDPOINTS ---

@app.get("/", tags=["Health Check"])
def health_check():
    """
    Verifica el estado de salud de la API y la versión del modelo.

    Returns:
        dict: Diccionario con estado 'online' y la versión del modelo.
    """
    return {"status": "online", "model_version": "1.1.0"}

@app.get(
    "/recommend/{user_id}",
    response_model=PredictResponse,
    tags=["Predictions"]
)
def get_recommendations(
    user_id: int,
    k: int = Query(5, gt=0, le=50, description="Número de recomendaciones a generar")
):
    """
    Genera una lista de K recomendaciones para un usuario basándose en su historial.

    El proceso sigue tres etapas:
    1. Filtrado de películas ya vistas por el usuario.
    2. Inferencia masiva sobre el catálogo restante usando el modelo NCF.
    3. Ordenamiento y recuperación de metadatos del Top-K resultados.

    Args:
        user_id (int): Identificador original del usuario.
        k (int): Cantidad de recomendaciones solicitadas. Valor entre 1 y 50.

    Returns:
        PredictResponse: Objeto con el ID del usuario y la lista de recomendaciones.

    Raises:
        HTTPException: 404 si el usuario no existe, 500 si ocurre un error en la predicción.
    """
    logger.info(f"Procesando solicitud de recomendación para User: {user_id} (K={k})")

    # Validación de existencia del usuario en el espacio de entrenamiento
    if user_id not in df['user_id'].unique():
        logger.warning(f"User ID {user_id} no presente en los datos originales.")
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    try:
        # Recuperación de índices internos y películas no vistas
        user_idx = df[df['user_id'] == user_id]['user_idx'].iloc[0]
        watched_movies = df[df['user_id'] == user_id]['movie_idx'].unique()
        all_movie_indices = df['movie_idx'].unique()
        candidate_movies = np.array([m for m in all_movie_indices if m not in watched_movies])

        # Preparación de inputs y ejecución de inferencia
        user_input = np.array([user_idx] * len(candidate_movies))
        predictions = model.predict([user_input, candidate_movies], verbose=0).flatten()

        # Selección de los mejores resultados
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

        logger.info(f"Pipeline finalizado. {len(results)} recomendaciones generadas.")
        return PredictResponse(user_id=user_id, recommendations=results)

    except Exception as e:
        logger.error(f"Error interno durante la predicción: {e}")
        raise HTTPException(status_code=500, detail="Error al generar recomendaciones.")