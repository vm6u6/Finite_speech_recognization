import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model_CNN import build_model
from model_CNN_LSTM import cnn_lstm
from data_preprocess import load_data, prepare_dataset, plot_history, lstm_data

def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history

def main():
    print("-- Loading Data... --")
    print()
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)

    # input_shape
    input_shape_CNN = (X_train.shape[1], X_train.shape[2], 1)
    input_shape_CNN = list(input_shape_CNN)
    print(input_shape_CNN)

    input_shape_LSTM = (X_train.shape[1], X_train.shape[2])
    input_shape_LSTM = list(input_shape_LSTM)
    print(input_shape_LSTM)
    
    # CNN 
    # model = build_model( input_shape_CNN, loss="sparse_categorical_crossentropy", learning_rate=0.0001 )
    # history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)
    # print( "[ Done!! ]" )
    # plot_history(history)                                 # plot accuracy/loss for training/validation set as a function of the epochs
    # test_loss, test_acc = model.evaluate(X_test, y_test)  # evaluate network on test set

    # CNN_LSTM
    model =  cnn_lstm ( input_shape_LSTM, loss="sparse_categorical_crossentropy", learning_rate=0.0001 )
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)
    print( "[ Done!! ]" )
    plot_history(history) 
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    # save model
    model.save(SAVED_MODEL_PATH)

DATA_PATH = "./data_LPC.json"
SAVED_MODEL_PATH = "./model_LPC.h5"
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 0.0001

if __name__ == "__main__":
    main()