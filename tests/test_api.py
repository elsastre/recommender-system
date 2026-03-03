from unittest.mock import patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# 1. Creamos DataFrames falsos (mocks) para que main.py pueda hacer sus cálculos sin romperse
mock_df = pd.DataFrame({
    'user_id': [1, 2],
    'movie_id': [101, 102],
    'rating': [4.5, 3.0],
    'user_idx': [0, 1],
    'movie_idx': [0, 1]
})

mock_movies = pd.DataFrame({
    'movie_id': [101, 102],
    'title': ['Mock Movie 1', 'Mock Movie 2'],
    'genres': ['Action', 'Comedy']
})

# 2. Parcheamos inyectando los DataFrames reales en lugar de MagicMocks vacíos
with patch('tensorflow.keras.models.load_model'), \
     patch('src.preprocess.DataProcessor.load_and_clean', return_value=mock_df), \
     patch('pandas.read_csv', return_value=mock_movies):
    from main import app

client = TestClient(app)

def test_read_main():
    """Prueba el punto de entrada raíz."""
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["status"]

def test_recommendation_user_not_found():
    """
    Prueba que la API maneja usuarios nuevos correctamente.
    Al no existir el usuario 999 en el mock_df, debe activarse el Cold Start.
    """
    response = client.get("/recommend/999?k=5")
    # El endpoint devuelve status 200 y activa la bandera is_cold_start
    assert response.status_code == 200
    assert response.json()["is_cold_start"] is True
