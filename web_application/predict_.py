import librosa
from scipy.io.wavfile import write
import sounddevice as sd
import wavio as wv
import tensorflow as tf
import numpy as np
import os
import pyttsx3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAVED_MODEL_PATH = "./model_LSTM_augment_check.h5"
SAMPLES_TO_CONSIDER = 22050
_mapping = ["down","go","left","no","off","on","right","stop","up","yes"]


def init_model():
    loaded_model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    print(" [Loaded Model Successfully] ")
    return loaded_model

def preprocess(file_path, num_mfcc=13, n_fft=2048, hop_length=512):

    # load audio file
    signal, sample_rate = librosa.load(file_path)

    if len(signal) >= SAMPLES_TO_CONSIDER:
        # ensure consistency of the length of the signal
        signal = signal[:SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
    return MFCCs.T

def prediction( model, file_path ):
    """
    :param file_path (str): Path to audio file to predict
    :return predicted_keyword (str): Keyword predicted by the model
    """

    # extract MFCC
    MFCCs = preprocess(file_path)

    # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

    # get the predicted label
    predictions = model.predict(MFCCs)
    predicted_index = np.argmax(predictions)
    predicted_keyword = _mapping[predicted_index]
    return predicted_keyword


# if __name__ == "__main__":
#     file_path = "test.wav"
#     model = init_model()
#     print()
#     print( "  [START PREDICTION]  " )
#     print( "----------------------" )
#     key_word = prediction( model, file_path )
#     print( "     Command:    ", key_word )