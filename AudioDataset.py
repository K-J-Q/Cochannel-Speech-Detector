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
import warnings
warnings.simplefilter("ignore")


def transformData(audio_paths, transformParams=None):
    """
    Outputs spectrogram in addition to any transforms indicated in transformParams (dictionary)

    audio_paths: List of .wav paths for dataset
    transformParams: List of dictionary with keys audio and spectrogram
    """
    transformedDataset = AudioDataset(audio_paths)
    if transformParams:

        for transform in transformParams:
            audio_train_dataset = AudioDataset(
                audio_paths,
                specTransformList=transform['spectrogram']
                if 'spectrogram' in transform else [],
                audioTransformList=transform['audio']
                if 'audio' in transform else [],
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
                 audioTransformList=None):

        self.specTransformList = specTransformList

        self.audioAugment = Compose(
            audioTransformList) if audioTransformList else None
        self.audio_paths = audio_paths

        self.Augmentor = Augmentor()
        self.config = ConfigParser().read('config.ini')

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        path, filename = os.path.split(self.audio_paths[idx])

        title, _ = os.path.splitext(filename)

        try:
            fsID, classID, occurrenceID, sliceID = [
                int(n) for n in title.split('-')
            ]
        except ValueError:
            classID = -1

        waveform, sample_rate = self.Augmentor.audio_preprocessing(
            torchaudio.load(self.audio_paths[idx]))
        
        if self.audioAugment:
            waveform = self.audioAugment(waveform.numpy(), sample_rate)
            if not torch.is_tensor(waveform):
                waveform = torch.from_numpy(waveform)

        waveform, sample_rate = self.Augmentor.pad_trunc(
            [waveform, sample_rate])

        spectrogram = torchaudio.transforms.Spectrogram()

        spectrogram_tensor = (spectrogram(waveform) + 1e-12).log2()

        # assert spectrogram_tensor.shape == torch.Size(
        #     [1, 201, 201]), f"Spectrogram size mismatch! {spectrogram_tensor.shape}"

        if self.specTransformList:

            for transform in self.specTransformList:

                spectrogram_tensor = transform(spectrogram_tensor)

        return [spectrogram_tensor, classID]


# TODO add __main__ as test function
