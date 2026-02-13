from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    """Test the root endpoint of the API."""
    response = client.get("/")
    assert response.status_code == 200
    assert "NCF Movie Recommender" in response.json()["message"]

def test_recommendation_endpoint():
    """Test if the recommendation endpoint returns a valid structure."""
    # Assuming user 1 exists in your data
    response = client.get("/recommend/1?k=5")
    assert response.status_code == 200
    assert "recommendations" in response.json()
    assert len(response.json()["recommendations"]) <= 5