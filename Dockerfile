# Dockerfile for the NCF recommendation service.
# Builds an image with Python 3.12, installs dependencies, and runs the FastAPI API.

# Use a lightweight Python 3.12 image
FROM python:3.12-slim

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

# Expose the port used by FastAPI
EXPOSE 8000

# Command to start the API when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]