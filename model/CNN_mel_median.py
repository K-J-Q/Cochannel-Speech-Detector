import torch_audiomentations as aug
from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from torchsummary import summary
from configparser import ConfigParser
import torchaudio
import sys


class CNNNetwork_mel_median(nn.Module):
    def __init__(self, nfft, outputClasses=3, normParam=8, augmentations=None):
        super(CNNNetwork_mel_median, self).__init__()

        self.audioNorm = aug.PeakNormalization(p=1)
        self.augmentor = augmentations

        self.generateSpec = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000, n_fft=nfft, n_mels=int(nfft / 4))

        self.normParam = normParam

        # self.augmentSpec = nn.Sequential(torchaudio.transforms.FrequencyMasking(
        #     30, True), torchaudio.transforms.TimeMasking(20, True))
        
        self.augmentSpec = torchvision.transforms.RandomVerticalFlip()

        self.conv1 = nn.Conv2d(1, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=1)
        self.conv7 = nn.Conv2d(1024, 2048, 3, stride=1)

        self.convBN1 = nn.LazyBatchNorm2d()
        self.convBN2 = nn.LazyBatchNorm2d()
        self.convBN3 = nn.LazyBatchNorm2d()
        self.convBN4 = nn.LazyBatchNorm2d()
        self.convBN5 = nn.LazyBatchNorm2d()
        self.convBN6 = nn.LazyBatchNorm2d()
        self.convBN7 = nn.LazyBatchNorm2d()

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm1d(120)
        self.fc1 = nn.LazyLinear(out_features=120, bias=False)
        self.fc2 = nn.Linear(120, outputClasses, bias=False)

    def __normaliseSpec(self, x):
        median_val = x.reshape(
            x.shape[0], -1).median(1).values.view(x.shape[0], 1, 1, 1)
        x = x/(x+self.normParam*median_val+1e-5)
        return x


    def __audioNormalisation(self, wav):
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

        if not self.training:
            spec = x

        

        x = self.convBN1(F.elu(self.conv1(x)))

        if x.shape[-1] >= 3 and x.shape[-2] >= 3:
            x = self.convBN2(F.elu(self.conv2(x)))
        if x.shape[-1] >= 3 and x.shape[-2] >= 3:
            x = self.convBN3(F.elu(self.conv3(x)))
        if x.shape[-1] >= 3 and x.shape[-2] >= 3:
            x = self.convBN4(F.elu(self.conv4(x)))
        if x.shape[-1] >= 3 and x.shape[-2] >= 3:
            x = self.convBN5(F.elu(self.conv5(x)))
        if x.shape[-1] >= 3 and x.shape[-2] >= 3:
            x = self.convBN6(F.elu(self.conv6(x)))
        if x.shape[-1] >= 3 and x.shape[-2] >= 3:
            x = self.convBN7(F.elu(self.conv7(x)))

        x = x.view(x.shape[0], -1)
        x = self.bn1(F.elu(self.fc1(x)))
        x = self.fc2(x)

        if self.training:
            return x

        return x, spec


def testModel():
    for nfft in [128, 256, 512, 1024, 2048]:
        for duration in [1000]:
            print(f"nfft: {nfft}, duration: {duration}")
            cnn = CNNNetwork_mel_whisper(nfft)
            sampleLength = int(8000*duration/1000)
            summary(cnn, (1, sampleLength), device="cpu")


if __name__ == "__main__":
    testModel()
    config = ConfigParser()
    config.read('config.ini')

    nfft = int(config['data']['n_fft'])
    duration = int(config['augmentations']['duration'])
    sampleLength = int(8000*duration/1000)

    cnn = CNNNetwork_mel(nfft)

    summary(cnn, (1, sampleLength), device="cpu")
    # print(cnn)
    print(cnn(torch.randn(2, 1, sampleLength)).shape)
    cnn.eval()
    print(cnn(torch.randn(2, 1, sampleLength)).shape)
