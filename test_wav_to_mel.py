import os
import argparse
import pickle
import glob
import random
import numpy as np
from tqdm import tqdm
import time

import librosa
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


wavpath = r"VOICEACTRESS100_001.wav"
sampling_rate = 24000
fft_size = 1024
num_mels = 80
fmin = 80
fmax = 7600
hop_size = 256
win_length = 1024
window = "hann"
eps=1e-10


mel_log_specs = None 


wav, _ = librosa.load(wavpath, sr=sampling_rate, mono=True)
print("wav shape : ",end="")
print(wav.shape)
print(type(wav))

s_time = time.time()

x_stft = librosa.stft(
    wav,
    n_fft=fft_size,
    hop_length=hop_size,
    win_length=win_length,
    window=window,
    pad_mode="reflect",
)
spc = np.abs(x_stft).T  # (#frames, #bins)

# get mel basis
fmin = 0 if fmin is None else fmin
fmax = sampling_rate / 2 if fmax is None else fmax
mel_basis = librosa.filters.mel(
    sr=sampling_rate,
    n_fft=fft_size,
    n_mels=num_mels,
    fmin=fmin,
    fmax=fmax,
)
mel = np.maximum(eps, np.dot(spc, mel_basis.T))

log_mel_spec = np.log10(mel).T



print("log mel spec shape : ",end="")
print(log_mel_spec.shape)

print("time : ", end="")
print(time.time()-s_time)

mel_mean = np.mean(log_mel_spec, axis=1, keepdims=True)
print("mel_mean",end="")
print(mel_mean.shape)

mel_std = np.std(log_mel_spec, axis=1, keepdims=True) + 1e-9
print("mel_std",end="")
print(mel_std.shape)



