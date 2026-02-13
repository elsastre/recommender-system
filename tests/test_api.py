from unittest.mock import patch, MagicMock
import pytest

# --- MOCKING: "Engañamos" a la API antes de que importe main.py ---
# Simulamos que el modelo y los datos se cargan bien aunque no existan los archivos
with patch('tensorflow.keras.models.load_model'), \
     patch('src.preprocess.DataProcessor.load_and_clean'), \
     patch('pandas.read_csv'):
    from main import app

from fastapi.testclient import TestClient
client = TestClient(app)

def test_read_main():
    """Prueba el punto de entrada raíz."""
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["status"]

def test_recommendation_user_not_found():
    """
    Prueba que la API maneja usuarios no encontrados.
    En este entorno, como el dataframe está mockeado (vacío), 
    cualquier usuario debería devolver 404.
    """
    response = client.get("/recommend/999?k=5")
    assert response.status_code == 404
