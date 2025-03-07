# # src/train.py
# import numpy as np
# import tensorflow as tf
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, f1_score

# from data_loader import load_and_preprocess_data
# from feature_engineering import add_economic_features, add_rolling_indicators, add_technical_indicators
# from model import create_mlp

# def extract_features(model, X):
#     feature_model = tf.keras.Model(
#         inputs=model.inputs,  # fix for the "Expected: ['keras_tensor']" issue
#         outputs=model.get_layer("feature_layer").output
#     )
#     return feature_model.predict(X)

# def main():
    
#     df = load_and_preprocess_data("data/raw/sp500_20years.csv")
    
#     df = add_economic_features(df)
#     df = add_rolling_indicators(df, window=14)
#     df = add_technical_indicators(df)

#     feature_cols = [
#     'Log_Returns', 
#     'LogReturns_Mean', 'LogReturns_Std', 'LogReturns_Skew', 'LogReturns_Kurtosis',
#     'RollingMean_14', 'RollingStd_14',
#     'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14'
#     ]
#     X = df[feature_cols].values
#     y = df['Target'].values
    
#     # 5. Time-based split (70% train, 15% val, 15% test)
#     n = len(df)
#     train_end = int(0.7 * n)
#     val_end = int(0.85 * n)
    
#     X_train = X[:train_end]
#     y_train = y[:train_end]
#     X_val = X[train_end:val_end]
#     y_val = y[train_end:val_end]
#     X_test = X[val_end:]
#     y_test = y[val_end:]
    
#     # 6. Build & compile the MLP
#     input_shape = (X_train.shape[1],)
#     mlp = create_mlp(input_shape)
#     mlp.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # 7. EarlyStopping callback to prevent overfitting
#     early_stopping = tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=5,           # stops if val_loss doesn't improve for 5 epochs
#         restore_best_weights=True
#     )
    
#     # 8. Train the MLP
#     mlp.fit(
#         X_train, y_train,
#         epochs=50,                 # max epochs
#         batch_size=32,
#         validation_data=(X_val, y_val),
#         callbacks=[early_stopping] # includes early stopping
#     )
    
#     # 9. Extract features from MLP for train & test
#     train_features = extract_features(mlp, X_train)
#     test_features = extract_features(mlp, X_test)
    
#     # 10. Train SVM on extracted features
#     svm = SVC(kernel='rbf', C=1.0, gamma='scale')
#     svm.fit(train_features, y_train)
    
#     # 11. Evaluate SVM on test set
#     y_pred = svm.predict(test_features)
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
    
#     print(f"SVM Classifier Accuracy: {accuracy:.4f}")
#     print(f"SVM Classifier F1 Score: {f1:.4f}")

# if __name__ == "__main__":
#     main()



import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from data_loader import load_and_preprocess_data
from feature_engineering import add_economic_features, add_rolling_indicators, add_technical_indicators
from model import create_lstm_model

def create_sequences(X, y, time_steps=10):
    """
    Converts an array of features and labels into overlapping sequences.
    :param X: Array of shape (samples, features)
    :param y: Array of target values (samples,)
    :param time_steps: Sequence length (number of days).
    :return: Tuple (X_seq, y_seq) where X_seq has shape (samples, time_steps, features)
             and y_seq has shape (samples,).
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def main():
    # 1. Load and preprocess data; processed data is saved in 'data/processed/'
    df = load_and_preprocess_data(
        raw_filepath="data/raw/sp500_20years.csv",
        processed_filepath="data/processed/sp500_20years_processed.csv"
    )
    
    # 2. Add economic features and technical indicators
    df = add_economic_features(df)
    df = add_rolling_indicators(df, window=14)
    df = add_technical_indicators(df)
    
    # 3. Define feature columns and target variable
    feature_cols = [
        'Log_Returns', 
        'LogReturns_Mean', 'LogReturns_Std', 'LogReturns_Skew', 'LogReturns_Kurtosis',
        'RollingMean_14', 'RollingStd_14',
        'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14'
    ]
    X = df[feature_cols].values
    y = df['Target'].values

    # 4. Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Convert daily features into sequences (using a 10-day window)
    time_steps = 10
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps=time_steps)
    
    # 6. Chronologically split the data: 70% train, 15% validation, 15% test
    n = len(X_seq)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    # 7. Build and compile the LSTM model with modified architecture
    input_shape = (time_steps, X_train.shape[2])
    lstm_model = create_lstm_model(input_shape, lstm_units=64, dropout_rate=0.3)
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 8. Add EarlyStopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # 9. Train the LSTM model
    lstm_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    # 10. Evaluate the model on the test set
    test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test)
    print("LSTM Test Accuracy: {:.4f}".format(test_accuracy))
    
if __name__ == "__main__":
    main()