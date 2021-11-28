import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"

def load_data( data_path ):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array( data["LPCs"] )
    y = np.array( data["labels"] )
    print( "Training sets loaded" )
    return X, y

def prepare_dataset( data_path, test_size = 0.2, validation_size = 0.2 ):
    X, y = load_data( data_path )
    
    # creat train, test, and validation
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = test_size )
    X_train, X_validation, y_train, y_validation = train_test_split( X_train, y_train, test_size=validation_size )
    
    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()
    plt.savefig('result.png')

def lstm_data( input_data ):
    X_train_lstm = []
    for i in range( len(input_data) ):
      X_train_j = []
      for j in range( len(input_data[0]) ):
        for z in range( len(input_data[0][0]) ):
          X_train_j.append( input_data[i][j][z][0] )
      X_train_lstm.append(X_train_j)
    return np.asarray(X_train_lstm)

def process():
    # generate train, validation and test sets
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)

if __name__ == "__main__":
    process()