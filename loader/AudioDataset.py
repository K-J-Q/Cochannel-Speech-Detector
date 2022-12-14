import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torch.nn.functional as F
import os
from torch.profiler import profile, record_function, ProfilerActivity
from audiomentations import Compose, RoomSimulator
from configparser import ConfigParser

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

    def __init__(self):
        pass

    def audio_preprocessing(self, audioIn):
        return self.resample(self.rechannel(audioIn))

    def pad_trunc(self, aud, reduce_only=False):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        target_len = int(sr/1000) * (self.audio_duration)

        # ((self.audio_duration-1000) if reduce_only else self.audio_duration)

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

    def rechannel(self, aud, showWarning=True):
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
        if showWarning:
            print('rechannel process triggered!')
        return ((resig, sr))

    def resample(self, aud, showWarning=True):
        sig, sr = aud
        if (sr == self.audio_sampling):
            # Nothing to do
            return aud
        if showWarning:
            print('resampling process triggered!')
        num_channels = sig.shape[0]

        resig = T.Resample(sr, self.audio_sampling)(sig)
        return ((resig, self.audio_sampling))


def createDataset(audio_paths, transformParams=[{}], outputAudio=False):
    combinedDataset = AudioDataset(
        audio_paths,
        outputAudio,
        specTransformList=transformParams[0]['spectrogram'] if 'spectrogram' in transformParams[0] else [
        ],
        audioTransformList=transformParams[0]['audio'] if 'audio' in transformParams[0] else [
        ],
        beforeCochannelListSox=transformParams[0]['before_cochannel_sox'] if 'before_cochannel_sox' in transformParams[0] else [
        ]
    )

    if transformParams:
        for transform in transformParams[1:]:
            audio_train_dataset = AudioDataset(
                audio_paths,
                outputAudio,
                specTransformList=transform['spectrogram']
                if 'spectrogram' in transform else [],
                audioTransformList=transform['audio']
                if 'audio' in transform else [],
                beforeCochannelListSox=transform['before_cochannel_sox']
                if 'before_cochannel' in transform else []
            )

            combinedDataset = torch.utils.data.ConcatDataset(
                [combinedDataset, audio_train_dataset])

    return combinedDataset


n_fft = int(config['data']['n_fft'])

# TODO: Remove outputAudio from AudioDataset (deprecated)

class AudioDataset(Dataset):
    """
    A custom dataset that fetches the audio path, load as waveform, perform augmentation audioTransformList and specTransformList and outputs the spectrogram
    audio_paths: List of .wav paths for dataset
    audioTransformList: audiomentations transforms
    specTransformList: pyTorch spectrogram masking options
    """

    class_size = int(config['data']['class_size'])
    windowLength = int(config['augmentations']['duration'])/1000

    augment = Compose([RoomSimulator()])

    def __init__(self,
                 audio_paths,
                 outputAudio,
                 specTransformList=None,
                 audioTransformList=None,
                 beforeCochannelListSox=None):

        self.specTransformList = specTransformList
        self.beforeCochannelAugmentSox = beforeCochannelListSox
        self.audioAugment = Compose(
            audioTransformList) if audioTransformList else None
        self.env_paths, self.speech_paths = audio_paths
        self.Augmentor = Augmentor()
        self.samplesPerClass = 3
        self.outputAudio = outputAudio
        self.dataShape = torch.Size([1, int(self.windowLength * 8000)])
        self.sampleLength = int(self.windowLength*8000)


    def __len__(self):
        return min(len(self.env_paths), len(self.speech_paths))

    def __getitem__(self, idx):
        env_aud = self.__getAudio(self.env_paths[idx])
        speech1_aud = self.__getAudio(self.speech_paths[idx])
        speech2_aud = self.__getAudio(
            self.speech_paths[random.randint(0, self.__len__()) - 1])

        X = torch.empty(
            [self.class_size*self.samplesPerClass] + list(self.dataShape))
        Y = torch.tensor([0, 1, 2]).repeat(self.class_size)

        for i in range(self.class_size):
            env = self.__split(env_aud)
            aud1 = self.__split(speech1_aud)
            aud2 = self.__split(speech2_aud)
            merged_aud = self.__merge_audio(aud1, aud2)

            X[self.samplesPerClass * i][0] = self.__augmentAudio(env)
            X[self.samplesPerClass*i+1][0] = self.__augmentAudio(aud1)
            X[self.samplesPerClass*i + 2][0] = self.__augmentAudio(merged_aud)
            # torchaudio.save('test.wav', X[self.samplesPerClass*i+2], 8000)
        return [X, Y]

    def __augmentAudio(self, wav, augments=[]):
        if 'add_noise' in augments:
            gain = random.uniform(0, 0.1)*wav.max()
            noise = torch.randn(wav.shape)
            wav = wav + gain * noise
        if 'reverb' in augments: 
            wav = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate=8000, effects=[["reverb", "70"], ['channels', '1']])[0]
        return wav
        

    def __removeDC(self, wav):
        return wav - wav.mean()

    def __split(self, audio):
        wav, sr = audio
        start_idx = random.randint(0, len(wav[0])-self.sampleLength)
        audio = wav[:, start_idx: start_idx+self.sampleLength]
        return audio

    def __normaliseAudio(self, wav):
        wav /= torch.max(torch.abs(wav))
        return wav

    def __merge_audio(self, aud1, aud2):
        gain_div = 0.2
        aud1 = self.__normaliseAudio(aud1)
        aud2 = self.__normaliseAudio(aud2)
        gain = random.uniform(0, gain_div)
        return aud1*(0.5-gain) + aud2*(0.5+gain)

    def __getAudio(self, audioPath):
        waveform, sample_rate = torchaudio.load(audioPath)
        # if self.beforeCochannelAugmentSox:
        #     audio, _ = torchaudio.sox_effects.apply_effects_tensor(
        #         torch.unsqueeze(audio, 0), 8000, self.beforeCochannelAugmentSox)
        waveform = self.__removeDC(waveform)
        return waveform, sample_rate


def collate_batch(batches):
    numBatch = len(batches)
    if (numBatch == 1):
        return batches[0][0], torch.Tensor(batches[0][1]).type(torch.LongTensor)

    samplesPerBatch = len(batches[0][0])

    X = torch.empty([samplesPerBatch * numBatch, 1, 16000])
    Y = torch.empty([samplesPerBatch * numBatch])

    for i, (x, y) in enumerate(batches):
        X[i*samplesPerBatch:(i+1)*samplesPerBatch] = x
        Y[i*samplesPerBatch:(i+1)*samplesPerBatch] = y

    Y = torch.Tensor(Y).type(torch.LongTensor)
    return X, Y


def main():
    import utils

    audio_path = utils.getAudioPaths('E:/Processed Audio/train/' if os.name == 'nt' else '/media/jianquan/Data/Processed Audio/train/')[0]
    dataset = createDataset(audio_path, utils.getTransforms(False), outputAudio=True)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_batch
    )
    for batch in iter(dataloader):
        pass


if __name__ == "__main__":
    main()
