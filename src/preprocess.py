"""Procesamiento de datos de MovieLens para sistema de recomendación NCF."""

import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Encargado del pipeline ETL y preprocesamiento de datos de MovieLens.

    Carga el dataset de calificaciones, genera índices continuos para usuarios
    y películas, y prepara los conjuntos de entrenamiento y prueba.

    Attributes:
        data_path (str): Ruta local al archivo de calificaciones (u.data).
        user_map (dict): Mapeo de índices internos a IDs de usuario originales.
        movie_map (dict): Mapeo de índices internos a IDs de película originales.
    """

    def __init__(self, data_path: str):
        """Inicializa el procesador con la ruta de los datos.

        Args:
            data_path (str): Ruta del archivo u.data de MovieLens.
        """
        self.data_path = data_path
        self.user_map = {}
        self.movie_map = {}

    def load_and_clean(self) -> pd.DataFrame:
        """Carga el dataset, realiza limpieza básica y genera mapeos de IDs.

        Lee el archivo tabulado, asigna nombres a las columnas, convierte los
        identificadores originales a índices numéricos continuos y almacena
        los mapeos para uso posterior.

        Returns:
            pd.DataFrame: DataFrame con las columnas originales y dos nuevas:
                'user_idx' e 'movie_idx', que contienen los índices internos.
        """
        names = ['user_id', 'movie_id', 'rating', 'timestamp']
        df = pd.read_csv(self.data_path, sep='\t', names=names)

        # Generar mapeos para evitar saltos en los índices de la red neuronal
        df['user_idx'] = df['user_id'].astype('category').cat.codes
        df['movie_idx'] = df['movie_id'].astype('category').cat.codes

        # Guardar mapeos para inferencia inversa
        self.user_map = dict(enumerate(df['user_id'].astype('category').cat.categories))
        self.movie_map = dict(enumerate(df['movie_id'].astype('category').cat.categories))

        return df

    def get_train_test(self, df: pd.DataFrame):
        """Divide el dataframe en conjuntos de entrenamiento y prueba.

        Extrae las características (user_idx, movie_idx) y la variable objetivo
        (rating normalizado). Realiza una partición estratificada simple.

        Args:
            df (pd.DataFrame): DataFrame procesado por `load_and_clean`.

        Returns:
            tuple: Contiene (X_train, X_test, y_train, y_test), donde:
                - X_train, X_test: arreglos con pares [user_idx, movie_idx].
                - y_train, y_test: arreglos con calificaciones normalizadas en [0,1].
        """
        X = df[['user_idx', 'movie_idx']].values
        y = (df['rating'].values - 1) / 4.0  # Normalización 0-1
        return train_test_split(X, y, test_size=0.2, random_state=42)