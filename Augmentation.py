import torch
import torchaudio
import random
from configparser import ConfigParser
from pathlib import Path
import os

config = ConfigParser()
config.read('config.ini')


class Augmentor():
    """
    Basic augmentation to ensure uniformaty among the different audio files.
    """
    audio_duration = int(config['augmentations']['duration'])
    audio_channels = int(config['augmentations']['num_channels'])
    audio_sampling = int(config['augmentations']['sample_rate'])
    noise_multiplier = float(config['augmentations']
                             ['pad_trunc_noise_multiplier'])

    def audio_preprocessing(self, audioIn):
        return self.pad_trunc(self.resample(self.rechannel(audioIn)), True)

    def pad_trunc(self, aud, reduce_only=False):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        target_len = int(sr/1000) * ((self.audio_duration-1000)
                                     if reduce_only else self.audio_duration)

        if (sig_len > target_len):
            start_len = random.randint(0, sig_len - target_len)
            sig = sig[:, start_len:start_len+target_len]
            assert (sig.shape[1] == target_len)

        elif (sig_len < target_len and not reduce_only):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, target_len - sig_len)
            pad_end_len = target_len - sig_len - pad_begin_len

            pad_begin = torch.rand(
                (num_rows, pad_begin_len))*self.noise_multiplier
            pad_end = torch.rand((num_rows, pad_end_len))*self.noise_multiplier
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    def rechannel(self, aud):
        sig, sr = aud
        if (sig.shape[0] == self.audio_channels):
            # Nothing to do
            return aud
        elif (self.audio_channels == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])
        print('rechannel process triggered!')
        return ((resig, sr))

    def resample(self, aud):
        sig, sr = aud
        if (sr == self.audio_sampling):
            # Nothing to do
            return aud
        print('resampling process triggered!')
        num_channels = sig.shape[0]
        # Resample first channel

        resig = torchaudio.transforms.Resample(sr,
                                               self.audio_sampling)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, self.audio_sampling)(
                sig[1:, :])
            resig = torch.cat([resig, retwo])
        return ((resig, self.audio_sampling))


def getAudioPaths(main_path, repeatMul=1):
    paths = list(Path(main_path).glob('**/*.wav'))
    for i in range(repeatMul):
        paths += paths
    return paths
