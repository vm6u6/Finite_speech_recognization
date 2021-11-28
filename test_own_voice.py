
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import tensorflow as tf
import numpy as np
import os
import pyttsx3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SAVED_MODEL_PATH = "./model/model_LSTM_augment_check.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    _mapping = ["down","go","left","no","off","on","right","stop","up","yes"]
    _instance = None

    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


def record_sound():

    # Sampling frequency
    freq = 22050
    # Recording duration
    duration = 1.25

    print( "--Start Recording--" )
    # Start recorder with the given values 
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    
    # Record audio for the given number of seconds
    sd.wait()

    print( "-- Recording End --" )

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("./test.wav", freq, recording)
    # Convert the NumPy array to audio file
    wv.write("./test.wav", recording, freq, sampwidth=2)
  

if __name__ == "__main__":

    record_sound()
    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()
    kss1 = Keyword_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1
    print()
    print( "[Start_Prediction]" )
    print()
    # make a prediction
    keyword = kss.predict("test.wav")
    print( keyword )

