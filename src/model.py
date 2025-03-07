import tensorflow as tf
from tensorflow.keras import layers, models

def create_mlp(input_shape, hidden_units=[64, 32]):
    """
    Builds a Multi-Layer Perceptron (MLP) for binary classification.
    The network includes a penultimate layer ("feature_layer") whose output will
    be used as features for an SVM classifier.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    for units in hidden_units:
        model.add(layers.Dense(units, activation='relu'))
    # Feature extractor layer
    model.add(layers.Dense(16, activation='relu', name='feature_layer'))
    # Final output layer: sigmoid activation for binary classification
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    return model