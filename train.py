"""
Training script for the NCF Recommendation model.

This module loads the MovieLens data, initializes the neural network
architecture, runs the training loop and saves the resulting model into
the 'models/' folder.
"""

from src.preprocess import DataProcessor
from src.model import create_ncf_model
import os

def main():
    """Run the full training pipeline and save the trained model."""
    # 1. Data preparation (ETL pipeline)
    processor = DataProcessor('data/u.data')
    df = processor.load_and_clean()
    X_train, X_test, y_train, y_test = processor.get_train_test(df)

    num_users = df['user_idx'].nunique()
    num_movies = df['movie_idx'].nunique()

    # 2. Initialize the NCF architecture
    model = create_ncf_model(num_users, num_movies)

    # 3. Training loop (optimization process)
    print("Starting training...")
    model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=64,
        epochs=10,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        verbose=1
    )

    # 4. Persist the artifact (model export)
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save('models/recommender_v1.keras')
    print("Model successfully saved to models/recommender_v1.h5")

if __name__ == "__main__":
    main()