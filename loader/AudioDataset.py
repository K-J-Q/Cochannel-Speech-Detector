import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
import loader.utils
from pathlib import Path
import torchaudio.transforms as T
import os
from torch.profiler import profile, record_function, ProfilerActivity
from audiomentations import Compose, RoomSimulator
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


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


def createDataset(audio_paths, generateCochannel=True, transformParams=[{}]):

    combinedDataset = __AudioDataset(
        audio_paths,
        specTransformList=transformParams[0]['spectrogram'] if 'spectrogram' in transformParams[0] else [
        ],
        audioTransformList=transformParams[0]['audio'] if 'audio' in transformParams[0] else [
        ],
        beforeCochannelListSox=transformParams[0]['before_cochannel_sox'] if 'before_cochannel_sox' in transformParams[0] else [
        ],
        generateCochannel=generateCochannel
    )

    if transformParams:
        for transform in transformParams[1:]:
            audio_train_dataset = __AudioDataset(
                audio_paths,
                specTransformList=transform['spectrogram']
                if 'spectrogram' in transform else [],
                audioTransformList=transform['audio']
                if 'audio' in transform else [],
                beforeCochannelListSox=transform['before_cochannel_sox']
                if 'before_cochannel' in transform else [],
                generateCochannel=generateCochannel
            )

            combinedDataset = torch.utils.data.ConcatDataset(
                [combinedDataset, audio_train_dataset])

    return combinedDataset


class __AudioDataset(Dataset):
    """
    A custom dataset that fetches the audio path, load as waveform, perform augmentation audioTransformList and specTransformList and outputs the spectrogram
    audio_paths: List of .wav paths for dataset
    audioTransformList: audiomentations transforms
    specTransformList: pyTorch spectrogram masking options
    """

    class_size = int(config['data']['class_size'])

    augment = Compose([RoomSimulator()])

    def __init__(self,
                 audio_paths,
                 specTransformList=None,
                 audioTransformList=None,
                 beforeCochannelListSox=None,
                 generateCochannel=True):

        self.specTransformList = specTransformList
        self.beforeCochannelAugmentSox = beforeCochannelListSox
        self.audioAugment = Compose(
            audioTransformList) if audioTransformList else None
        self.env_paths, self.speech_paths = audio_paths
        self.Augmentor = Augmentor()
        self.generateCochannel = generateCochannel
        
        if generateCochannel:
            self.specPerClass = 3
        else:
            self.specPerClass = 2

    def __len__(self):
        return min(len(self.env_paths), len(self.speech_paths))

    def __getitem__(self, idx):
        env_aud = self.__getAudio(self.env_paths[idx])
        speech1_aud = self.__getAudio(self.speech_paths[idx])
        assert env_aud[1] == speech1_aud[1]

        if self.generateCochannel:
            speech2_aud = self.__getAudio(
                self.speech_paths[random.randint(0, self.__len__()) - 1])
            assert speech1_aud[1] == speech2_aud[1]

        X = torch.zeros([self.class_size*self.specPerClass, 1,  201, 161])
        Y = []

        spectrogram = torchaudio.transforms.Spectrogram(
            normalized=True)
        
        for i in range(self.class_size):
            X[self.specPerClass *
                i][0] = (spectrogram(self.__split(env_aud)) + 1e-12).log2()
            X[self.specPerClass*i+1][0] = (spectrogram(self.__split(speech1_aud)) + 1e-12).log2()
            if self.generateCochannel:
                aud1 = self.__split(speech1_aud)
                aud2 = self.__split(speech2_aud)
                merged_aud = self.__merge_audio(aud1, aud2)
                # torchaudio.save(loader.utils.uniquify(
                #     './merged.wav'), merged_audio, 8000)
                # torchaudio.save(loader.utils.uniquify(
                #     './oldmerged.wav'), (aud1 + aud2) / 2, 8000)
                X[self.specPerClass*i +
                    2][0] = (spectrogram(merged_aud) + 1e-12).log2()
                Y.extend([0, 1, 2])
            else:
                Y.extend([0, 1])

        return [X, Y]

    def __split(self, audio, duration=4):
        cut_length = duration*audio[1]
        start_idx = random.randint(0, len(audio[0][0])-cut_length)
        audio = audio[0][0][start_idx: start_idx+cut_length]
        if self.beforeCochannelAugmentSox:
            audio, _ = torchaudio.sox_effects.apply_effects_tensor(torch.unsqueeze(audio,0), 8000, self.beforeCochannelAugmentSox)
        return audio

    def __merge_audio(self, aud1, aud2):
        a_pos = aud1*(aud1>=0)
        b_pos = aud2*(aud2>=0)
        pos = a_pos*(a_pos > b_pos) + b_pos*(b_pos>a_pos)

        a_neg = aud1*(aud1<0)
        b_neg = aud2*(aud2<0)
        neg = a_neg*(a_neg < b_neg) + b_neg*(b_neg<a_neg)

        return pos+neg

    def __getAudio(self, audioPath):
        waveform, sample_rate = self.Augmentor.audio_preprocessing(
            torchaudio.load(audioPath))
        # if self.beforeCochannelAugment:
        #     waveform = self.beforeCochannelAugment(
        #         waveform.numpy(), sample_rate)
        #     if not torch.is_tensor(waveform):
        #         waveform = torch.from_numpy(waveform)
        return waveform, sample_rate


def specMask(spectrogram):
    fMasking = T.FrequencyMasking(freq_mask_param=30)
    tMasking = T.TimeMasking(time_mask_param=30)
    return(tMasking(fMasking(spectrogram)))


def collate_batch(batches):

    if (len(batches) == 1):
        return batches[0][0], torch.Tensor(batches[0][1]).type(torch.LongTensor)
    X = torch.empty(0)
    Y = []

    for x, y in batches:
        X = torch.cat((X, x))
        Y.extend(y)
    Y = torch.Tensor(Y).type(torch.LongTensor)
    return X, Y


def main():
    import utils
    env_paths = utils.getAudioPaths('E:/Processed Audio/ENV/4 Diff Room')
    speech_paths = utils.getAudioPaths(
        'E:/Processed Audio/SPEECH/4 Diff Room')

    dataset = createDataset(env_paths, speech_paths,
                            utils.getTransforms(False))

    dataloader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_batch
    )
    for batch in iter(dataloader):
        pass


if __name__ == "__main__":
    main()
