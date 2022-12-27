import torch_audiomentations as aug
from torch import nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from configparser import ConfigParser
import torchaudio
import sys

config = ConfigParser()
config.read('config.ini')


class CNNNetwork_mel(nn.Module):
    def __init__(self, nfft, augmentations=None):
        super(CNNNetwork_mel, self).__init__()

        self.audioNorm = aug.PeakNormalization(p=1)
        self.augmentor = augmentations

        self.generateSpec = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000, n_fft=nfft)

        # self.augmentSpec = nn.Sequential(torchaudio.transforms.FrequencyMasking(
        #     30, True), torchaudio.transforms.TimeMasking(20, True))

        self.conv1 = nn.Conv2d(1, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=3)
        self.conv4 = nn.Conv2d(64, 128, 5, stride=3)
        self.conv5 = nn.Conv2d(128, 256, 5, stride=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drp = nn.Dropout2d(0)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc1 = nn.LazyLinear(out_features=120, bias=False)
        self.fc2 = nn.Linear(120, 3, bias=False)

    def __normaliseSpec(self, x):
        # normalisation method 1: whisper
        x = torch.clamp(x, min=1e-10).log10()
        max_val = x.reshape(
            x.shape[0], 1, -1).amax(2).view(x.shape[0], 1, 1, 1)
        x = torch.maximum(x, max_val - 8)

        # normalisation method 2: min 0, max 1
        min_val = x.reshape(
            x.shape[0], 1, -1).amin(2).view(x.shape[0], 1, 1, 1)
        x -= min_val
        max_val = x.reshape(
            x.shape[0], 1, -1).amax(2).view(x.shape[0], 1, 1, 1)
        x /= max_val

        # normalisation method 3: mean 0, std 1
        x = (x - x.mean()) / x.std()
        return x

        # return x/(x+10*x.median()+1e-12)

    def __audioNormalisation(self, wav):
        if isinstance(wav, torch.Tensor):
            wav = self.audioNorm.train()(wav)
            if self.augmentor is not None:
                wav = self.augmentor(wav, 8000)
        return wav

    def forward(self, wav):
        wav = self.__audioNormalisation(wav)
        x = self.generateSpec(wav)
        x = self.__normaliseSpec(x)

        # if self.training:
        #     x = self.augmentSpec(x)

        x = self.drp(F.elu(self.conv1(x)))
        x = self.drp(F.elu(self.conv2(x)))
        x = self.drp(F.elu(self.conv3(x)))
        # x = self.drp(F.elu(self.conv4(x)))
        # x = self.drp(F.elu(self.conv5(x)))
        x = x.view(x.shape[0], -1)
        x = self.bn1(F.elu(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    cnn = CNNNetwork_mel(512)

    summary(cnn, (1, 16000), device="cpu")
    # print(cnn)
    print(cnn(torch.randn(2, 1, 16000)).shape)
    cnn.eval()
    print(cnn(torch.randn(2, 1, 16000)).shape)
