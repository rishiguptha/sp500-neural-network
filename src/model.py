from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.25, learning_rate=0.001):
    """
    Builds and compiles an LSTM model with two LSTM layers and dropout.

    Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features).
        lstm_units (int): Number of units for each LSTM layer.
        dropout_rate (float): Dropout rate applied after each LSTM layer.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (tf.keras.Model): Compiled LSTM model.
    """
    model = Sequential()
    # First LSTM layer returns sequences so that we can stack another LSTM on top.
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape, 
                  kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer does not return sequences.
    model.add(LSTM(lstm_units, kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate))
    
    # Dense output layer (adjust activation and units according to your problem).
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with appropriate loss and optimizer.
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Example usage:
    # Suppose your LSTM expects sequences with 10 timesteps and 11 features.
    input_shape = (10, 11)
    model = build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.3)
    model.summary()