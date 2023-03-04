#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import threading

import librosa
from librosa.filters import mel as librosa_mel_fn

import torch
import torch.utils.data as data
import torchaudio

from parallel_wavegan.datasets import MelDataset
from parallel_wavegan.datasets import MelSCPDataset
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5

from mask_cyclegan_vc.model import Generator
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver


class VC:
    def __init__(self, sampling_rate, speaker_A_id, speaker_B_id, 
                    preprocessed_data_dir, ckpt_dir, model_name, load_epoch,
                    fft_size=1024, num_mels=80, fmax=7600, fmin=80, window="hann"):
                                
        self.device = torch.device("cpu")
        self.args = self.parser(sampling_rate, speaker_A_id, speaker_B_id, 
                                preprocessed_data_dir, ckpt_dir, model_name, load_epoch
                                )
        self.sampling_rate = sampling_rate
        # parameter for STFT
        self.fft_size = fft_size
        self.hop_size = fft_size // 4
        self.win_length = fft_size
        self.window = window
        self.num_mels = num_mels
        self.fmax = fmax
        self.fmin = fmin
        
        # arguments for model
        outdir = r"result"
        config = r"vocoder\conf\config.yml"
        checkpoint = r"vocoder\exp\checkpoint-2500000steps.pkl"
        
        # model setup
        self.vocoder = None
        self.cyclegan = None
        self.setup_vocoder(outdir, config, checkpoint)
        self.setup_cyclegan(self.args)
        
        
    def inference(self, source_wav):
        spec,_ , _ = self.wav_to_mel(source_wav)
        mel = torch.unsqueeze(torch.from_numpy(spec).clone(), dim=0)
        source_mel = mel.to(self.device, dtype=torch.float)
        
        print("source mel rmse : ",end="")
        print(np.sqrt(np.square(source_mel).mean()))
        #print("source mel shape : ", end="")
        #print(source_mel.shape)
        target_mel = self.cyclegan(source_mel, torch.ones_like(source_mel))
        #print("target mel shape : ", end="")
        #print(target_mel.shape)
        print("target mel rmse : ",end="")
        print(np.sqrt(np.square(target_mel.to('cpu').detach().numpy().copy()).mean()))
        target_wav = self.vocoder.inference(target_mel).view(-1)
        print("target wav rmse : ",end="")
        print(np.sqrt(np.square(target_wav.to('cpu').detach().numpy().copy()).mean()))
        
        return target_wav
        
    def setup_cyclegan(self, args):
        # Generator
        self.cyclegan = Generator().to(self.device)
        self.cyclegan.eval()
        
        model_name = args.model_name
        # Load Generator from ckpt
        self.saver = ModelSaver(args)
        self.saver.load_model(self.cyclegan, model_name)
        
    def setup_vocoder(self, outdir, config, checkpoint):
        # check directory existence
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        # load config
        if config is None:
            dirname = os.path.dirname(checkpoint)
            config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        
        conf_dict = {
                    "checkpoint": checkpoint,
                    "feats_scp": None,
                    "normalize_before": False,
                    "outdir": outdir
                    }
        config.update(conf_dict)
        
        # setup model
        self.vocoder = load_model(checkpoint, config)
        logging.info(f"Loaded model parameters from {checkpoint}.")
        self.vocoder.remove_weight_norm()
        self.vocoder = self.vocoder.eval().to(self.device)
        
    def wav_to_mel(self, wav, eps=1e-10):
        x_stft = librosa.stft(
            wav,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            pad_mode="reflect",
        )
        spc = np.abs(x_stft).T  # (#frames, #bins)

        # get mel basis
        mel_basis = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.fft_size,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        print("spc rmse : ",end="")
        print(np.sqrt(np.square(wav).mean()))
        mel = np.maximum(eps, np.dot(spc, mel_basis.T))

        log_mel_spec = np.log10(mel).T
        mel_mean = np.mean(log_mel_spec, axis=1, keepdims=True)
        mel_std = np.std(log_mel_spec, axis=1, keepdims=True) + 1e-9
        mel_norm = (log_mel_spec - mel_mean) / mel_std
        
        return mel_norm, mel_mean, mel_std
    
    
    def parser(self, sampling_rate, speaker_A_id, speaker_B_id, 
                    preprocessed_data_dir, ckpt_dir, model_name, load_epoch):
                    
        parser = argparse.ArgumentParser(description=' ')
        parser.add_argument(
                '--sample_rate', type=int, default=sampling_rate, 
                help='Sampling rate of mel-spectrograms.'
                )
        parser.add_argument(
                '--speaker_A_id', type=str, default=speaker_A_id, 
                help='Source speaker id (From VOC dataset).'
                )
        parser.add_argument(
                '--speaker_B_id', type=str, default=speaker_B_id, 
                help='Source speaker id (From VOC dataset).'
                )
        parser.add_argument(
                '--preprocessed_data_dir', type=str, default=preprocessed_data_dir, 
                help='Directory containing preprocessed dataset files.'
                )
        parser.add_argument(
                '--ckpt_dir', type=str, default=ckpt_dir, help='Path to model ckpt.')
        parser.add_argument(
                '--model_name', type=str, choices=('generator_A2B', 'generator_B2A'), 
                default=model_name, help='Name of model to load.'
                )
        parser.add_argument(
                '--load_epoch', type=str, default=load_epoch, 
                help='load epoch number.'
                )
        args = parser.parse_args()
        
        return args




def thread1_inference(model, wav):
    start_time = time.time()
    target_wav = model.inference(wav)
    conv_time = time.time() - start_time
    wav_time = target_wav.shape[0] / 24000.0
    #print("RTF = ",end="")
    #print(conv_time / wav_time)

if __name__ ==  "__main__":
    sampling_rate = 24000
    speaker_A_id = "me"
    speaker_B_id = "jvs063"
    preprocessed_data_dir = r"preprocessed_data"
    ckpt_dir = r"maskcyclegan_result"
    model_name = "generator_A2B"
    load_epoch = 5300
    
    wavpath = r"VOICEACTRESS100_001.wav"
    wav, _ = librosa.load(wavpath, sr=sampling_rate, mono=True)
    print(wav.shape)
    
    VC_test = VC(sampling_rate, speaker_A_id, speaker_B_id, 
                preprocessed_data_dir, ckpt_dir, model_name, load_epoch)
    
    #start_time = time.time()
    """
    target_wav = VC_test.inference(wav)
    
    conv_time = time.time() - start_time
    wav_time = target_wav.shape[0] / 24000.0
    print("RTF = ",end="")
    print(conv_time / wav_time)
    
    print("source_wav shape : ",end="")
    print(wav.shape)
    print("target wav shape : ",end="")
    print(target_wav.shape)
    sf.write(
            "result/target.wav",
            target_wav.detach().numpy(),
            sampling_rate,
            "PCM_16",
            )
    """

    t1 = threading.Thread(target=thread1_inference, args=(VC_test, wav))
    t1.start()

