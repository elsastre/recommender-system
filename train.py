import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
from src.preprocess import DataProcessor
from src.model import create_ncf_model
from tensorflow.keras.callbacks import EarlyStopping

def main():
    # 1. Data preparation (ETL pipeline)
    processor = DataProcessor('data/ratings.dat')
    df = processor.load_and_clean()
    X_train, X_test, y_train, y_test = processor.get_train_test(df)

    num_users = df['user_idx'].nunique()
    num_movies = df['movie_idx'].nunique()

    # 2. Initialize the NCF architecture
    model = create_ncf_model(num_users, num_movies)

    # 3. Training loop (optimization process)
    print("Starting training...")
    
    # Configure Early Stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=64,
        epochs=20,
        validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # 4. Persist the artifact (model export) and training history
    if not os.path.exists('models'):
        os.makedirs('models')

    model.save('models/recommender_v1.keras')
    print("Model successfully saved to models/recommender_v1.keras")

    history_dict = {
        'loss': [float(i) for i in history.history['loss']],
        'val_loss': [float(i) for i in history.history['val_loss']],
        'mae': [float(i) for i in history.history['mae']],
        'val_mae': [float(i) for i in history.history['val_mae']]
    }
    
    with open('models/training_history.json', 'w') as f:
        json.dump(history_dict, f)
    print("Training history saved to models/training_history.json")

if __name__ == "__main__":
    main()