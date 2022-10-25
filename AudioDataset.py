import torch
import torchaudio
import random

from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import os

from Augmentation import Augmentor

import numpy as np
from audiomentations import Compose
from configparser import ConfigParser


def transformData(audio_paths, generateCochannel, transformParams=None):
    """
    Outputs spectrogram in addition to any transforms indicated in transformParams (dictionary)

    audio_paths: List of .wav paths for dataset
    transformParams: List of dictionary with keys audio and spectrogram
    """
    transformedDataset = AudioDataset(
        audio_paths, generateCochannel=generateCochannel)

    if transformParams:
        for transform in transformParams:
            audio_train_dataset = AudioDataset(
                audio_paths,
                specTransformList=transform['spectrogram']
                if 'spectrogram' in transform else [],
                audioTransformList=transform['audio']
                if 'audio' in transform else [],
                beforeCochannelList=transform['before_cochannel']
                if 'before_cochannel' in transform else [],
            )

            transformedDataset = torch.utils.data.ConcatDataset(
                [transformedDataset, audio_train_dataset])

    return transformedDataset


class AudioDataset(Dataset):
    """
    A custom dataset that fetches the audio path, load as waveform, perform augmentation audioTransformList and specTransformList and outputs the spectrogram
    audio_paths: List of .wav paths for dataset
    audioTransformList: audiomentations transforms
    specTransformList: pyTorch spectrogram masking options
    """

    def __init__(self,
                 audio_paths,
                 specTransformList=None,
                 audioTransformList=None,
                 beforeCochannelList=None,
                 generateCochannel=False):
        
        self.specTransformList = specTransformList
        self.beforeCochannelAugment = Compose(
            beforeCochannelList) if beforeCochannelList else None
        self.audioAugment = Compose(
            audioTransformList) if audioTransformList else None
        self.audio_paths = audio_paths

        self.Augmentor = Augmentor()
        self.generateCochannel = generateCochannel
        self.config = ConfigParser().read('config.ini')

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        generateCochannel = random.randint(
            1, 2) if self.generateCochannel else 1

        path = os.path.basename(os.path.split(self.audio_paths[idx])[0])

        if path == "ENV":
            generateCochannel = 1

        combinedWaveform = torch.zeros([1, 5*8000])
        for i in range(generateCochannel):
            waveform, sample_rate = self.Augmentor.audio_preprocessing(
                torchaudio.load(self.audio_paths[idx-i]))

            if  self.beforeCochannelAugment:
                waveform = self.beforeCochannelAugment(
                    waveform.numpy(), sample_rate)
                if not torch.is_tensor(waveform):
                    waveform = torch.from_numpy(waveform)

            waveform, sample_rate = self.Augmentor.pad_trunc(
                [waveform, sample_rate])

            combinedWaveform += waveform

        if self.audioAugment:
            combinedWaveform = self.audioAugment(
                combinedWaveform.numpy(), sample_rate)
            if not torch.is_tensor(combinedWaveform):
                combinedWaveform = torch.from_numpy(combinedWaveform)

        spectrogram = torchaudio.transforms.Spectrogram()
        spectrogram_tensor = (spectrogram(combinedWaveform/2) + 1e-12).log2()

        assert spectrogram_tensor.shape == torch.Size(
            [1, 201, 201]), f"Spectrogram size mismatch! {spectrogram_tensor.shape}"

        if self.specTransformList:
            for transform in self.specTransformList:
                spectrogram_tensor = transform(spectrogram_tensor)

        if path == "ENV":
            return [spectrogram_tensor, 0]
        else:
            return [spectrogram_tensor, generateCochannel]


# TODO add __main__ as test function
