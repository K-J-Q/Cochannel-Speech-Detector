import torch
import torchaudio
import random
import utils
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import os

from Augmentation import Augmentor

import numpy as np
from audiomentations import Compose
from configparser import ConfigParser


def transformData(audio_paths, generateCochannel=True, transformParams=None):
    """
    Outputs spectrogram in addition to any transforms indicated in transformParams (dictionary)

    audio_paths: List of .wav paths for dataset
    transformParams: List of dictionary with keys audio and spectrogram
    """
    combinedDataset = AudioDataset(
        audio_paths,
        specTransformList=transformParams[0]['spectrogram'] if 'spectrogram' in transformParams[0] else [
        ],
        audioTransformList=transformParams[0]['audio'] if 'audio' in transformParams[0] else [
        ],
        beforeCochannelList=transformParams[0]['before_cochannel'] if 'before_cochannel' in transformParams[0] else [
        ],
        generateCochannel=generateCochannel
    )

    if transformParams:
        for transform in transformParams[1:]:
            audio_train_dataset = AudioDataset(
                audio_paths,
                specTransformList=transform['spectrogram']
                if 'spectrogram' in transform else [],
                audioTransformList=transform['audio']
                if 'audio' in transform else [],
                beforeCochannelList=transform['before_cochannel']
                if 'before_cochannel' in transform else [],
                generateCochannel=generateCochannel
            )

            combinedDataset = torch.utils.data.ConcatDataset(
                [combinedDataset, audio_train_dataset])

    return combinedDataset


class AudioDataset(Dataset):
    """
    A custom dataset that fetches the audio path, load as waveform, perform augmentation audioTransformList and specTransformList and outputs the spectrogram
    audio_paths: List of .wav paths for dataset
    audioTransformList: audiomentations transforms
    specTransformList: pyTorch spectrogram masking options
    """

    envPath = Path('E:/Processed Audio/ENV')

    def __init__(self,
                 audio_paths,
                 specTransformList=None,
                 audioTransformList=None,
                 beforeCochannelList=None,
                 generateCochannel=True):

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

        num_speakers = random.randint(
            1, 2) if self.generateCochannel else 1

        aud_source = self.audio_paths[idx].parents[1]

        if aud_source == self.envPath:
            num_speakers = 0

        combinedWaveform = torch.zeros([1, 5*8000])
        waveform, sample_rate = self.__getAudio(idx)
        combinedWaveform += waveform

        for i in range(num_speakers - 1):
            combinedWaveform += self.__getAudio(
                random.randint(0, self.__len__()) - 1)[0]

        # for testing audio only [REMOVE BEFORE TRAINING]
        # if num_speakers == 2:
        #     torchaudio.save(utils.uniquify('../testfile.wav'),
        #                     combinedWaveform, sample_rate)

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

        if aud_source == "ENV":
            return [spectrogram_tensor, 0]

        else:
            return [spectrogram_tensor, num_speakers]

    def __getAudio(self, index):
        waveform, sample_rate = self.Augmentor.audio_preprocessing(
            torchaudio.load(self.audio_paths[index]))

        if self.beforeCochannelAugment:
            waveform = self.beforeCochannelAugment(
                waveform.numpy(), sample_rate)
            if not torch.is_tensor(waveform):
                waveform = torch.from_numpy(waveform)

        waveform, sample_rate = self.Augmentor.pad_trunc(
            [waveform, sample_rate])

        return waveform, sample_rate

# TODO add __main__ as test function
