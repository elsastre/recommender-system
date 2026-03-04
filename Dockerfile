# Dockerfile for the NCF recommendation service.
# Builds an image with Python 3.12, installs dependencies, and runs the FastAPI API.

# Use a lightweight Python 3.12 image
FROM tensorflow/tensorflow:2.18.0

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for HDF5 and TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code
COPY . .

# Forzamos a TensorFlow a usar solo CPU para que no pierda tiempo buscando drivers
ENV CUDA_VISIBLE_DEVICES="-1"

# Usamos el formato shell para que Render pueda inyectar su propio puerto dinámico
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}