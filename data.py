import numpy as np
import librosa
import os 
import json
from utils import is_bad_audio, lpc_coeff

DATASET_PATH = "./data/train_valerio"
JSON_PATH = "data.json"
JSON_PATH1 = "data_LPC.json"
JSON_PATH2 = "data_MFCCLPC.json"
SAMPLES_TO_CONSIDER = 22050

def preprocess_dataset(dataset_path, json_path, json_path1, json_path2, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    data_LPC = {
        "mapping": [],
        "labels": [],
        "LPCs": [],
        "files": []
    }

    data_MFCCLPC = {
        "mapping": [],
        "labels": [],
        "LPCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            data_LPC["mapping"].append(label)
            data_MFCCLPC["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file and slice it to ensure length consistency among different files
                signal, sample_rate = librosa.load(file_path)

                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # check quality
                    if is_bad_audio(signal): 
                        # ensure consistency of the length of the signal
                        signal = signal[:SAMPLES_TO_CONSIDER]
                        print("{}: {}".format(file_path, i-1))

                        # LPC
                        LPCs = []
                        L = SAMPLES_TO_CONSIDER//1600
                        for j in range( L ):
                            frame = signal[j*1600:((j+1)*1600)-1]
                            coeff, res = lpc_coeff(frame, 10)
                            if res == False:
                                break
                            LPCs.append(coeff.tolist())
                        LPCs = np.array(LPCs)
                        if res == True:
    
                            # MFCC
                            # MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                            #                             hop_length=hop_length)
                            
                            # store data for analysed track
                            # data["MFCCs"].append(MFCCs.T.tolist())
                            # data["labels"].append(i-1)
                            # data["files"].append(file_path)

                            data_LPC["LPCs"].append(LPCs.tolist())
                            data_LPC["labels"].append(i-1)
                            data_LPC["files"].append(file_path)

                            # data_MFCCLPC["LPCs"].append(MFCCs.T.tolist() + LPCs.T.tolist())
                            # data_MFCCLPC["labels"].append(i-1)
                            # data_MFCCLPC["files"].append(file_path)


    # save data in json file
    # with open(json_path, "w") as fp:
    #     json.dump(data, fp, indent=4)
    with open(json_path1, "w") as fp:
        json.dump(data_LPC, fp, indent=4)
    # with open(json_path2, "w") as fp:
    #     json.dump(data_MFCCLPC, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH, JSON_PATH1, JSON_PATH2,)