import tensorflow as tf
from keras import backend

backend.set_image_data_format('channels_last')

def cnn_lstm(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # define CNN model
    model.add(tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    
    # define LSTM model
    model.add(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True))
    model.add(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=False))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model
