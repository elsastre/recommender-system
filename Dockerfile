# Dockerfile para el servicio de recomendación NCF.
# Construye una imagen con Python 3.12, instala dependencias y ejecuta la API FastAPI.

# Usamos una imagen ligera de Python 3.12
FROM python:3.12-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos dependencias del sistema necesarias para HDF5 y TensorFlow
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos primero los requerimientos para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código del proyecto
COPY . .

# Exponemos el puerto que usa FastAPI
EXPOSE 8000

# Comando para arrancar la API cuando el contenedor inicie
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]