import numpy as np 
import librosa
import os
import cv2

#------------------------------ data augmentation ---------------------------------------
# wav is the signal
# sr is the sample rate
# 22050 is the SAMPLES_TO_CONSIDER

def time_shifting( wav, sr ):
    start_ = int(np.random.uniform(-4800,4800))
    if start_ >= 0:
        wav_time_shift = np.r_[wav[start_:], np.random.uniform(-0.001,0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), wav[:start_]]
    return wav_time_shift

def speed_tuning( wav  ):
    speed_rate = np.random.uniform(0.7, 1.3)
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
    if len(wav_speed_tune) < 22050:
        pad_len = 22050 - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
                            wav_speed_tune,
                            np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
    else: 
        cut_len = len(wav_speed_tune) - 22050
        wav_speed_tune = wav_speed_tune[int(cut_len/2):int(cut_len/2)+22050]
    return wav_speed_tune

def background_noise( wav ):
    bg_files = os.listdir('./data/white_noise')
    chosen_bg_file = bg_files[np.random.randint(2)]
    bg, sr = librosa.load('./data/white_noise/'+chosen_bg_file, sr=None)
    start_ = np.random.randint(bg.shape[0]-22050)
    bg_slice = bg[start_ : start_+22050]
    wav_with_bg = wav * np.random.uniform(0.8, 1.2) + \
                bg_slice * np.random.uniform(0, 0.1)
    return wav_with_bg

#------------------------------------ check qulity ---------------------------------------------

def is_bad_audio( signal ):
    features = librosa.feature.spectral_centroid( signal, sr=22050, n_fft=2048, hop_length=512 )
    m = np.mean(features)
    t = np.std(features)
    if  t < 80:
        # silent
        return False
    elif (m > 2550 and t < 300):
        # noisy
        return False
    elif (m > 3500 and t > 1200):
        # distorted
        return False
    else:
        # good
        return True

#------------------------------------ LPC Coeff ---------------------------------------------

def lpc_coeff(s, p):
    n = len(s)
    ar = np.zeros(p + 1)
    Rp = np.zeros(p)
    for i in range(p):
        Rp[i] = np.sum(np.multiply(s[i + 1:n], s[:n - i - 1]))
    Rp0 = np.matmul(s, s.T)
    if Rp0 == 0.0:
        print(Rp0)
        return ar, False
    Ep = np.zeros((p, 1))
    k = np.zeros((p, 1))
    a = np.zeros((p, p))

    Ep0 = Rp0
    k[0] = Rp[0] / Rp0
    a[0, 0] = k[0]
    Ep[0] = (1 - k[0] * k[0]) * Ep0

    if p > 1:
        for i in range(1, p):
            k[i] = (Rp[i] - np.sum(np.multiply(a[:i, i - 1], Rp[i - 1::-1]))) / Ep[i - 1]
            a[i, i] = k[i]
            Ep[i] = (1 - k[i] * k[i]) * Ep[i - 1]
            for j in range(i - 1, -1, -1):
                a[j, i] = a[j, i - 1] - k[i] * a[i - j - 1, i - 1]
    
    ar[0] = 1
    ar[1:] = -a[:, p - 1]
    return ar, True
