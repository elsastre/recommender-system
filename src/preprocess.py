"""MovieLens data processing module for NCF systems."""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


class DataProcessor:
    """
    Handles the ETL pipeline and preprocessing for the MovieLens dataset.

    Attributes:
        data_path (str): Path to the raw u.data file.
        user_map (Dict): Mapping from internal indices to original user IDs.
        movie_map (Dict): Mapping from internal indices to original movie IDs.
    """

    def __init__(self, data_path: str):
        """
        Initialize the processor.

        Args:
            data_path (str): Local path to the ratings file.
        """
        self.data_path = data_path
        self.user_map: Dict = {}
        self.movie_map: Dict = {}

    def load_and_clean(self) -> pd.DataFrame:
        """
        Load dataset and generate contiguous category indices.

        Returns:
            pd.DataFrame: Processed dataframe with 'user_idx' and 'movie_idx'.
        """
        names = ['user_id', 'movie_id', 'rating', 'timestamp']
        df = pd.read_csv(self.data_path, sep='\t', names=names)

        # Vectorized category encoding
        df['user_idx'] = df['user_id'].astype('category').cat.codes
        df['movie_idx'] = df['movie_id'].astype('category').cat.codes

        # Store mappings for inference lookup
        self.user_map = dict(enumerate(df['user_id'].astype('category').cat.categories))
        self.movie_map = dict(enumerate(df['movie_id'].astype('category').cat.categories))

        return df

    def get_train_test(self, df: pd.DataFrame) -> Tuple:
        """
        Split data into training and testing sets with normalized targets.

        Args:
            df (pd.DataFrame): Dataframe from load_and_clean().

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        X = df[['user_idx', 'movie_idx']].values
        y = (df['rating'].values - 1) / 4.0  # Normalización 0-1
        return train_test_split(X, y, test_size=0.2, random_state=42)