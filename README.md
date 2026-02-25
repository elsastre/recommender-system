d![Python CI](https://github.com/elsastre/recommender-system/actions/workflows/python-app.yml/badge.svg)

# 🎬 Neural Collaborative Filtering (NCF) Movie Recommender

A production-ready recommendation system based on **Deep Learning**. This project implements an NCF architecture to predict user preferences and serves them through a scalable **FastAPI** service containerized with **Docker**.

## 🚀 Key Features
- **Deep Learning Architecture**: Built with TensorFlow/Keras using Embedding layers and Multi-Layer Perceptron (MLP).
- **Production-Grade API**: Robust backend with data validation (Pydantic) and structured logging.
- **Containerized Environment**: Fully dockerized for consistent deployment across any system.
- **Interactive UI**: User-friendly dashboard built with Streamlit for real-time recommendations.

## 🧠 Architecture
The system uses **Neural Collaborative Filtering**. Instead of simple matrix factorization, it uses a neural network to learn the non-linear interaction between users and items.

$$y_{ui} = \sigma(MLP(P^T v_u \oplus Q^T v_i))$$

![NCF Architecture](./docs/ncf-architecture.png)

- **Input**: User IDs and Movie IDs.
- **Latent Space**: High-dimensional Embeddings.
- **Interaction Layer**: Concatenated vectors passed through Dense layers with ReLU activation.
- **Output**: A probability score (0-1) representing the likelihood of interest.

## 🛠️ Tech Stack
- **Engine**: Python 3.12, TensorFlow, Pandas, NumPy.
- **API**: FastAPI, Uvicorn, Pydantic.
- **DevOps**: Docker.
- **Frontend**: Streamlit.

Here are the **final, copyable instructions** for your `README.md`. All commands are single‑line and tested for PowerShell on Windows (they also work on Linux/macOS). No more syntax errors.

---

## 📦 Getting Started (One‑Line Commands)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed.
- Download the [MovieLens 100k dataset](https://files.grouplens.org/datasets/movielens/ml-100k.zip) and place `u.data` and `u.item` inside the `data/` folder.

### 1. Build the Docker image
```bash
docker build -t movie-recommender-api .
```

### 2. Train the model (inside the container)
This creates `models/recommender_v1.keras` on your host.
```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" movie-recommender-api python train.py
```

### 3. Start the API service
The API will be available at `http://localhost:8000`.
```bash
docker run --name recommender-service -p 8000:8000 -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" movie-recommender-api
```

### 4. Run the Streamlit UI (on your host)
Open a **new terminal** and run:
```bash
pip install streamlit
python -m streamlit run src/app_ui.py
```
The UI will open at `http://localhost:8501` and connect to the running API.

---

### ⚠️ Important Notes
- The `models` folder will be created automatically after training.
- If you close the API container, restart it with:
  ```bash
  docker start recommender-service
  ```
- To stop the API container: `docker stop recommender-service`

## 📊 Model Performance
The model was trained for 10 epochs. The best generalization was observed around **Epoch 4**, before the onset of overfitting.

| Metric | Value |
| :--- | :--- |
| Training Loss (MSE) | 0.0480 |
| Validation Loss (MSE) | 0.0540 |
| Validation MAE | 0.1829 |

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

*Note: Ratings are normalized to a [0, 1] scale. An MAE of 0.18 on a 5-star scale represents an average error of approximately 0.9 stars.*

Developed by Braihans - AI Engineering Student
