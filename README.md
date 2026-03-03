# 🎬 AIFlix: Neural Collaborative Filtering (NCF) Recommender

![Python CI](https://github.com/elsastre/recommender-system/actions/workflows/python-app.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

A production-ready recommendation system based on **Deep Learning**. This project implements an NCF architecture to predict user preferences and serves them through a scalable **FastAPI** backend, fully containerized with **Docker**, and visualized via a **Streamlit** enterprise-grade dashboard.

## 🚀 Key Features

* **Deep Learning Architecture**: Replaces traditional matrix factorization with a neural network (Embedding layers + MLP) to capture non-linear user-item interactions.
* **Cold Start Policy**: Implements a global baseline fallback for new users without historical data.
* **Vector Similarity Search**: Computes item-to-item recommendations using Cosine Similarity on the learned embedding latent space.
* **Production-Grade API**: Robust backend with data validation (Pydantic), asynchronous request handling, and dynamic TMDB metadata fetching.
* **Microservices Orchestration**: Fully dockerized backend and frontend communicating over an internal Docker network.

## 🧠 Architecture & Methodology

The core engine utilizes **Neural Collaborative Filtering**. The model learns dual high-dimensional embeddings for users and items, concatenates them, and passes the vector through a dense Multi-Layer Perceptron with ReLU activations.



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
* A [TMDB API Key](https://developer.themoviedb.org/docs/getting-started) for fetching movie posters.

### Installation

1. Clone the repository and navigate to the project root:
```bash
git clone [https://github.com/elsastre/recommender-system.git](https://github.com/elsastre/recommender-system.git)
cd recommender-system
