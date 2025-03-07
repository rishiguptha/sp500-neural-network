import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import tensorflow as tf
import math
from model import build_lstm_model

def load_data(file_path='data/processed/sp500_20years_processed.csv'):
    """
    Load the processed CSV file and return features and target.
    Adjust this function based on your CSV structure.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)  # ensure data is in time order

    # Assume 'Target' is the label and all other numeric columns (except Date) are features.
    X = df.drop(['Date', 'Target'], axis=1).values
    y = df['Target'].values
    return X, y

def create_sequences(features, target, timesteps=10):
    """
    Convert the flat feature array into sequences for LSTM.
    Each sequence will have shape (timesteps, num_features).
    """
    X_seq, y_seq = [], []
    for i in range(len(features) - timesteps):
        X_seq.append(features[i:i+timesteps])
        y_seq.append(target[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

def standardize_features(features):
    """
    Standardize features using z-score normalization (mean=0, std=1).
    """
    # Calculate mean and standard deviation for each feature
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    # Handle any features with zero standard deviation to avoid division by zero
    std[std == 0] = 1
    
    # Standardize the features: (X - mean) / std
    standardized_features = (features - mean) / std
    
    return standardized_features

def main():
    # Load data
    X, y = load_data()

    # Standardize the features (z-score normalization)
    X_scaled = standardize_features(X)

    # Set the number of timesteps (look-back period)
    timesteps = 10
    X_seq, y_seq = create_sequences(X_scaled, y, timesteps)

    # Split data into train and test sets (here we use a time-based split so we don't shuffle)
    split_index = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_index], X_seq[split_index:]
    y_train, y_test = y_seq[:split_index], y_seq[split_index:]

    # Build the LSTM model:
    # Here the input shape is (timesteps, num_features)
    num_features = X_seq.shape[2]
    model = build_lstm_model(input_shape=(timesteps, num_features),
                             lstm_units=64,       # use 64 units as in original model
                             dropout_rate=0.25,   # moderate dropout rate
                             learning_rate=0.001) # standard learning rate

    model.summary()

    # Set up callbacks for training
    # Early stopping to prevent unnecessary epochs
    early_stop = EarlyStopping(
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Combine all callbacks
    callbacks = [early_stop, reduce_lr]
    
    # Calculate class weights if the dataset is imbalanced
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=100,                  # extend training to 100 epochs maximum
                        batch_size=32,               # standard batch size
                        validation_split=0.1,
                        callbacks=callbacks,         # use the callbacks defined above
                        class_weight=class_weights_dict,  # handle class imbalance
                        verbose=1)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

if __name__ == "__main__":
    main()