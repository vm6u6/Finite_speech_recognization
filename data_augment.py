import librosa
import os 
import json
from utils import time_shifting, speed_tuning, background_noise, is_bad_audio


DATASET_PATH = "./data/train_valerio"
JSON_PATH = "data_augment_check.json"
SAMPLES_TO_CONSIDER = 22050

def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):

    data = { "mapping": [], "labels": [], "MFCCs": [], "files": [] }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    
                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # check quality
                    if is_bad_audio(signal):

                        # data augumentation
                        signal_time_shifting = time_shifting(signal, sample_rate)
                        signal_speed_tuning = speed_tuning( signal )
                        signal_backgroud_noise = background_noise( signal )

                        # extract MFCCs
                        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                    hop_length=hop_length)
                        MFCCs_time = librosa.feature.mfcc(signal_time_shifting, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                    hop_length=hop_length)
                        MFCCs_speed = librosa.feature.mfcc(signal_speed_tuning, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                    hop_length=hop_length)
                        MFCCs_backgroud = librosa.feature.mfcc(signal_backgroud_noise, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                    hop_length=hop_length)

                        # store data for analysed track
                        data["MFCCs"].append(MFCCs.T.tolist())
                        data["labels"].append(i-1)
                        data["files"].append(file_path)
                        data["MFCCs"].append(MFCCs_time.T.tolist())
                        data["labels"].append(i-1)
                        data["files"].append(file_path)
                        data["MFCCs"].append(MFCCs_speed.T.tolist())
                        data["labels"].append(i-1)
                        data["files"].append(file_path)
                        data["MFCCs"].append(MFCCs_backgroud.T.tolist())
                        data["labels"].append(i-1)
                        data["files"].append(file_path)
                    
    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)