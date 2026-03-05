# 🎬 Neural Collaborative Filtering (NCF) Recommender

![Python CI](https://github.com/elsastre/recommender-system/actions/workflows/python-app.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

A production-ready recommendation system based on **Deep Learning**. This project implements an NCF architecture to predict user preferences and serves them through a scalable **FastAPI** backend, fully containerized with **Docker**, and visualized via a **Streamlit** enterprise-grade dashboard.

## 🚀 Key Features

* **Deep Learning Architecture**: Replaces traditional matrix factorization with a neural network (Embedding layers + MLP) to capture non-linear user-item interactions.
* **Microservices Orchestration**: Fully dockerized backend and frontend communicating over an internal Docker network, ensuring 100% local reproducibility.
* **Cold Start Policy**: Implements a global baseline fallback for new users without historical data to prevent API failures.
* **Vector Similarity Search**: Computes item-to-item recommendations using Cosine Similarity on the learned embedding latent space.
* **Production-Grade API**: Robust backend with data validation (Pydantic), asynchronous request handling, deferred asset loading, and dynamic TMDB metadata fetching.

## 🧠 Architecture & Methodology

The core engine utilizes **Neural Collaborative Filtering**. The model learns dual high-dimensional embeddings for users and items, concatenates them, and passes the vector through a dense Multi-Layer Perceptron (MLP) with ReLU activations.

The interaction probability is calculated as:
$$y_{ui} = \sigma(MLP(P^T v_u \oplus Q^T v_i))$$

* **Latent Space**: Dynamic extraction of weights from the Keras embedding layers for similarity computation.
* **Optimization**: Trained with Adam optimizer and Early Stopping to prevent overfitting.

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Engine** | Python 3.12, TensorFlow / Keras, Scikit-Learn, Pandas |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Streamlit |
| **DevOps & CI** | Docker, Docker Compose, GitHub Actions (Flake8, Pytest) |

## 📦 Getting Started (Local Deployment)

The entire infrastructure is orchestrated via Docker Compose, making deployment seamless.

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed.
* Python 3.12+ (for the initial asset download script).
* A [TMDB API Key](https://developer.themoviedb.org/docs/getting-started) for fetching movie posters.

### Installation

1. **Clone the repository** and navigate to the project root:
```bash
git clone [https://github.com/elsastre/recommender-system.git](https://github.com/elsastre/recommender-system.git)
cd recommender-system
```

2. **Configure environment variables**:
Create a `.env` file in the root directory and add your TMDB key:
```env
TMDB_API_KEY=your_api_key_here
```

3. **Download artifacts**:
Run the setup script to automatically fetch the MovieLens 1M dataset and the pre-trained `.keras` model (this keeps the Git repository lightweight):
```bash
python download_assets.py
```

4. **Spin up the microservices**:
Build and start the containers in detached mode:
```bash
docker-compose up --build -d
```

### Accessing the Services
* **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
* **FastAPI Interactive Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📊 Model Performance

The model generalizes effectively on the MovieLens 1M dataset. 

| Metric | Value |
| :--- | :--- |
| Training Loss (MSE) | 0.0480 |
| Validation Loss (MSE) | 0.0540 |
| Validation MAE | 0.1829 |

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

*Note: Ratings are normalized to a [0, 1] scale. An MAE of 0.18 represents an average error of approximately 0.9 stars on a standard 5-star scale.*

---
*Developed by Braihans - AI Engineering Student*