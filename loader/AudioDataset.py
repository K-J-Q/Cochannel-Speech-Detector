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

    def __init__(self):
        pass

    def audio_preprocessing(self, audioIn):
        return self.resample(self.rechannel(audioIn))

    def pad_trunc(self, aud, reduce_only=False):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        target_len = int(sr / 1000) * (self.audio_duration)

        # ((self.audio_duration-1000) if reduce_only else self.audio_duration)

        if (sig_len > target_len):
            start_len = random.randint(0, sig_len - target_len)
            sig = sig[:, start_len:start_len + target_len]
            assert (sig.shape[1] == target_len)

        elif sig_len < target_len and not reduce_only:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, target_len - sig_len)
            pad_end_len = target_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return sig, sr

    def rechannel(self, aud, showWarning=True):
        sig, sr = aud
        if sig.shape[0] == self.audio_channels:
            # Nothing to do
            return aud
        elif self.audio_channels == 1:
            # Convert from stereo to mono by selecting only the first channel
            resignal = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resignal = torch.cat([sig, sig])
        if showWarning:
            print('rechannel process triggered!')
        return resignal, sr

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


n_fft = int(config['data']['n_fft'])


class AudioDataset(Dataset):
    class_size = int(config['data']['class_size'])
    windowLength = int(config['augmentations']['duration']) / 1000

    add_noise = float(config['augmentations']['augment_noise'])
    gain_div = float(config['augmentations']['gain_div'])

    def __init__(self,
                 audio_paths,
                 outputAudio,
                 isTraining,
                 add_noise=0,
                 gain_div=0,
                 num_merge=2,
                 mergePercentage=(1, 1)):

        self.isTraining = isTraining
        if not isTraining:
            self.class_size = 25

        self.env_paths, self.speech_paths = audio_paths

        self.Augmentor = Augmentor()
        self.samplesPerClass = num_merge + 1

        self.outputAudio = outputAudio
        self.dataShape = torch.Size([1, int(self.windowLength * 8000)])
        self.sampleLength = int(self.windowLength * 8000)
        self.add_noise = add_noise
        self.gain_div = gain_div

        self.mergePercentage = mergePercentage if mergePercentage[0] < mergePercentage[1] else None


    def __len__(self):
        return min(len(self.env_paths), len(self.speech_paths))

    def __getitem__(self, idx):
        speech0_aud = self.__getAudio(self.env_paths[idx])
        speech1_aud = self.__getAudio(self.speech_paths[idx])
        speech2_aud = self.__getAudio(self.speech_paths[random.randint(0, self.__len__()) - 1])
        if self.samplesPerClass == 4:
            speech3_aud = self.__getAudio(self.speech_paths[random.randint(0, self.__len__()) - 1])

        X = torch.empty([self.class_size * self.samplesPerClass] + list(self.dataShape))
        Y = torch.tensor(list(range(self.samplesPerClass))).repeat(self.class_size)

        for i in range(self.class_size):
            env = self.__split(speech0_aud)
            aud1 = self.__split(speech1_aud)
            aud2 = self.__split(speech2_aud)
            merged_aud = self.__merge_audio(aud1, aud2)
            X[self.samplesPerClass * i][0] = self.__augmentAudio(env)
            X[self.samplesPerClass * i + 1][0] = self.__augmentAudio(aud1)
            X[self.samplesPerClass * i + 2][0] = self.__augmentAudio(merged_aud)
            if self.samplesPerClass == 4:
                aud3 = self.__split(speech3_aud)
                merged_aud = self.__merge_audio(aud1, aud2, aud3)
                X[self.samplesPerClass * i + 3][0] = self.__augmentAudio(merged_aud)

            # torchaudio.save('test.wav', X[self.samplesPerClass*i+3], 8000)

        return [X, Y]

    def __augmentAudio(self, wav, augments=[]):
        if self.isTraining:
            sampleRate = 8000
            if 'highpass' in augments:
                wav = torchaudio.functional.highpass_biquad(wav, sampleRate, 50)
            if 'add_noise' in augments and self.add_noise > 0:
                gain = random.uniform(0, self.add_noise)
                noise = torch.randn(wav.shape)
                wav = wav + gain * noise
            if 'reverb' in augments:
                wav = torchaudio.sox_effects.apply_effects_tensor(wav, sample_rate=sampleRate,
                                                                  effects=[["reverb", "70"], ['channels', '1']])[0]
        return wav

    def __removeDC(self, wav):
        return wav - wav.mean()

    def __normaliseAudio(self, wav):
        wav /= torch.max(torch.abs(wav))
        return wav

    def __split(self, audio):
        wav, sr = audio
        start_idx = random.randint(0, len(wav[0]) - self.sampleLength)
        audio = wav[:, start_idx: start_idx + self.sampleLength]
        return audio

    def __merge_audio(self, *auds):
        merged_aud = torch.zeros(auds[0].shape)
        if self.mergePercentage is None:
            audioLength = len(auds[0][0])
        else:
            audioLength = int(random.uniform(self.mergePercentage[0], self.mergePercentage[1])*len(auds[0][0]))

        for i, aud in enumerate(auds):
            gain = 1
            aud = self.__normaliseAudio(aud)
            if self.gain_div:
                gain = 1 - random.uniform(-self.gain_div, self.gain_div)
            aud = aud[:, :audioLength] if i else aud
            # torchaudio.save(f'aud{i}.wav', self.__normaliseAudio(aud), 8000)
            merged_aud[:, -len(aud[0]):] += aud * gain
        # torchaudio.save('merged.wav', merged_aud, 8000)
        return merged_aud

    def __getAudio(self, audioPath):
        waveform, sample_rate = torchaudio.load(audioPath)
        waveform = self.__removeDC(waveform)
        return waveform, sample_rate


def collate_batch(batches):
    numBatch = len(batches)
    # get a single sample
    audioLength = len(batches[0][0][0][0])

    if (numBatch == 1):
        return batches[0][0], torch.Tensor(batches[0][1]).type(torch.LongTensor)

    samplesPerBatch = len(batches[0][0])

    X = torch.empty([samplesPerBatch * numBatch, 1, audioLength])
    Y = torch.empty([samplesPerBatch * numBatch])

    for i, (x, y) in enumerate(batches):
        X[i * samplesPerBatch:(i + 1) * samplesPerBatch] = x
        Y[i * samplesPerBatch:(i + 1) * samplesPerBatch] = y

    Y = torch.Tensor(Y).type(torch.LongTensor)
    return X, Y


def main():
    import utils

    audio_path = utils.getAudioPaths(
        'E:/Processed Audio/train/' if os.name == 'nt' else '/media/jianquan/Data/Processed Audio/train/')[0]

    dataset = AudioDataset(audio_path, outputAudio=True, isTraining=True)

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
