import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path='data/processed/sp500_20years_processed.csv'):
    """
    Load the processed CSV file and return features and target.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)  # ensure data is in time order

    # Assume 'Target' is the label and all other numeric columns (except Date) are features.
    X = df.drop(['Date', 'Target'], axis=1).values
    y = df['Target'].values
    return X, y, df.drop(['Date', 'Target'], axis=1).columns

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

def create_sliding_window_features(features, target, window_size=10):
    """
    Create sliding window features where each row uses the last N days of features
    to predict the next day's direction.
    
    Args:
        features: Array of feature values, shape (num_samples, num_features)
        target: Array of target values, shape (num_samples,)
        window_size: Number of previous days to use as features
        
    Returns:
        X_windowed: Windowed features, shape (num_samples - window_size, window_size * num_features)
        y_windowed: Target values corresponding to the windowed features
    """
    num_samples, num_features = features.shape
    X_windowed = np.zeros((num_samples - window_size, window_size * num_features))
    y_windowed = np.zeros(num_samples - window_size)
    
    for i in range(num_samples - window_size):
        # Flatten the window of features
        X_windowed[i, :] = features[i:i+window_size, :].flatten()
        # The target is the value after the window
        y_windowed[i] = target[i+window_size]
    
    return X_windowed, y_windowed

def plot_feature_importance(feature_importances, feature_names, top_n=20):
    """
    Plot the feature importances from the Random Forest model.
    
    Args:
        feature_importances: Array of feature importance values
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    # Create a DataFrame for easy sorting and plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Select top N features
    importance_df = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

def main():
    # Load data
    X, y, feature_names = load_data()
    
    # Standardize the features
    X_scaled = standardize_features(X)
    
    # Create sliding window features
    window_size = 10  # Same as the sequence length in the LSTM model
    X_windowed, y_windowed = create_sliding_window_features(X_scaled, y, window_size)
    
    # Create new feature names for the windowed features
    windowed_feature_names = []
    for i in range(window_size):
        for feature in feature_names:
            windowed_feature_names.append(f"{feature}_lag{i+1}")
    
    # Split data into train and test sets (using time-based split)
    split_index = int(0.8 * len(X_windowed))
    X_train, X_test = X_windowed[:split_index], X_windowed[split_index:]
    y_train, y_test = y_windowed[:split_index], y_windowed[split_index:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize the Random Forest model with specified parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    # Cross-validation evaluation
    print("Performing cross-validation...")
    cv_results = cross_validate(
        rf_model, 
        X_train, 
        y_train, 
        cv=5,
        scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
        return_train_score=True
    )
    
    # Print cross-validation results
    print("\nCross-Validation Results:")
    print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"CV Precision: {cv_results['test_precision_weighted'].mean():.4f} ± {cv_results['test_precision_weighted'].std():.4f}")
    print(f"CV Recall: {cv_results['test_recall_weighted'].mean():.4f} ± {cv_results['test_recall_weighted'].std():.4f}")
    print(f"CV F1 Score: {cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}")
    
    # Train the model on the full training set
    print("\nTraining the final model...")
    rf_model.fit(X_train, y_train)
    
    # Evaluate on the test set
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print metrics
    print("\nTest Set Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Feature importance analysis
    print("\nFeature Importance Analysis:")
    feature_importances = rf_model.feature_importances_
    plot_feature_importance(feature_importances, windowed_feature_names)
    
    # Print top 10 most important features
    importance_df = pd.DataFrame({
        'feature': windowed_feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))

if __name__ == "__main__":
    main()

