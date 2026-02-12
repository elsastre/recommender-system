"""MovieLens data processing for an NCF recommender system."""

import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Responsible for the ETL pipeline and preprocessing of MovieLens data.

    Loads the ratings dataset, creates contiguous indices for users and movies,
    and prepares train/test splits.

    Attributes:
        data_path (str): Local path to the ratings file (u.data).
        user_map (dict): Mapping from internal indices to original user IDs.
        movie_map (dict): Mapping from internal indices to original movie IDs.
    """

    def __init__(self, data_path: str):
        """Initialize the processor with the data path.

        Args:
            data_path (str): Path to the MovieLens u.data file.
        """
        self.data_path = data_path
        self.user_map = {}
        self.movie_map = {}

    def load_and_clean(self) -> pd.DataFrame:
        """Load the dataset, perform basic cleaning and create ID mappings.

        Reads the tab-separated file, assigns column names, converts original
        identifiers to continuous numeric indices and stores the mappings for
        later use.

        Returns:
            pd.DataFrame: DataFrame with the original columns and two new ones:
                'user_idx' and 'movie_idx' containing internal indices.
        """
        names = ['user_id', 'movie_id', 'rating', 'timestamp']
        df = pd.read_csv(self.data_path, sep='\t', names=names)

        # Generate mappings to avoid gaps in neural network indices
        df['user_idx'] = df['user_id'].astype('category').cat.codes
        df['movie_idx'] = df['movie_id'].astype('category').cat.codes

        # Save mappings for inverse lookup during inference
        self.user_map = dict(enumerate(df['user_id'].astype('category').cat.categories))
        self.movie_map = dict(enumerate(df['movie_id'].astype('category').cat.categories))

        return df

    def get_train_test(self, df: pd.DataFrame):
        """Split the dataframe into training and testing sets.

        Extracts features (user_idx, movie_idx) and the target variable
        (normalized rating). Performs a simple train/test split.

        Args:
            df (pd.DataFrame): DataFrame processed by `load_and_clean`.

        Returns:
            tuple: Contains (X_train, X_test, y_train, y_test) where:
                - X_train, X_test: arrays with pairs [user_idx, movie_idx].
                - y_train, y_test: arrays with normalized ratings in [0,1].
        """
        X = df[['user_idx', 'movie_idx']].values
        y = (df['rating'].values - 1) / 4.0  # Normalización 0-1
        return train_test_split(X, y, test_size=0.2, random_state=42)