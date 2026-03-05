from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import app

# 1. Creamos DataFrames falsos (mocks) para no depender de los CSV reales
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

# Mock del modelo para que la extracción de embeddings en el startup no tire error
mock_model = MagicMock()
mock_model.layers = []

# 2. Inyectamos los mocks y levantamos el cliente de pruebas
@pytest.fixture
def client():
    """
    Fixture que levanta el TestClient.
    Mantiene los parches activos durante toda la prueba.
    """
    with patch('tensorflow.keras.models.load_model', return_value=mock_model), \
         patch('src.preprocess.DataProcessor.load_and_clean', return_value=mock_df), \
         patch('pandas.read_csv', return_value=mock_movies):
        
        # CLAVE: Usar TestClient dentro de un 'with' obliga a FastAPI 
        # a ejecutar los eventos de @app.on_event("startup")
        with TestClient(app) as test_client:
            yield test_client

# 3. Pruebas

def test_read_main(client):
    """Prueba el punto de entrada raíz."""
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["status"]

def test_recommendation_user_not_found(client):
    """
    Prueba que la API maneja usuarios nuevos correctamente.
    Al no existir el usuario 999 en el mock_df, debe activarse el Cold Start.
    """
    response = client.get("/recommend/999?k=5")
    # El endpoint debe devolver 200 y activar la bandera is_cold_start
    assert response.status_code == 200
    assert response.json()["is_cold_start"] is True