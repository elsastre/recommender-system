"""
Script de entrenamiento para el modelo de Recomendación NCF.

Este módulo carga los datos de MovieLens, inicializa la arquitectura de la red
neuronal, ejecuta el ciclo de entrenamiento y persiste el modelo resultante
en la carpeta 'models/'.
"""

from src.preprocess import DataProcessor
from src.model import create_ncf_model
import os

def main():
    """Ejecuta el pipeline completo de entrenamiento y guardado del modelo."""
    # 1. Preparación de datos (Pipeline ETL)
    processor = DataProcessor('data/u.data')
    df = processor.load_and_clean()
    X_train, X_test, y_train, y_test = processor.get_train_test(df)

    num_users = df['user_idx'].nunique()
    num_movies = df['movie_idx'].nunique()

    # 2. Inicialización de la arquitectura NCF
    model = create_ncf_model(num_users, num_movies)

    # 3. Ciclo de entrenamiento (Optimization process)
    print("Iniciando entrenamiento...")
    model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=64,
        epochs=10,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        verbose=1
    )

    # 4. Persistencia del artefacto (Model Export)
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save('models/recommender_v1.h5')
    print("Modelo guardado exitosamente en models/recommender_v1.h5")

if __name__ == "__main__":
    main()