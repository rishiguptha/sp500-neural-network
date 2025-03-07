import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from data_loader import load_and_preprocess_data
from feature_engineering import add_economic_features
from model import create_mlp

def extract_features(model, X):
    """
    Extracts features from the penultimate layer ('feature_layer') of the MLP.
    """
    feature_model = tf.keras.Model(inputs=model.inputs,
                                   outputs=model.get_layer("feature_layer").output)
    features = feature_model.predict(X)
    return features

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("data/raw/sp500_20years.csv")
    df = add_economic_features(df)
    
    # Define feature columns and target variable
    feature_cols = ['Log_Returns', 'LogReturns_Mean', 'LogReturns_Std', 'LogReturns_Skew', 'LogReturns_Kurtosis']
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Split data into training, validation, and test sets using time-based split
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    # Build and compile the MLP model
    input_shape = (X_train.shape[1],)
    mlp = create_mlp(input_shape)
    mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    # Train the MLP
    mlp.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    
    # Extract features from the penultimate layer for training and test sets
    train_features = extract_features(mlp, X_train)
    test_features = extract_features(mlp, X_test)
    
    # Train an SVM classifier on the extracted features
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(train_features, y_train)
    
    # Evaluate the SVM classifier on the test set
    y_pred = svm.predict(test_features)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("SVM Classifier Accuracy: {:.4f}".format(accuracy))
    print("SVM Classifier F1 Score: {:.4f}".format(f1))
    
if __name__ == "__main__":
    main()