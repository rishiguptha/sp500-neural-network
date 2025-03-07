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

def create_lstm_model(input_shape, lstm_units=64, dropout_rate=0.3):
    """
    Builds an LSTM model for binary classification using two LSTM layers and dropout.
    :param input_shape: Tuple (timesteps, features)
    :param lstm_units: Number of units in each LSTM layer.
    :param dropout_rate: Dropout rate for regularization.
    :return: A compiled Keras model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

