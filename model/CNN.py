from torch import nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from configparser import ConfigParser
import torchaudio

config = ConfigParser()
config.read('config.ini')


class CNNNetwork(nn.Module):
    def __init__(self, nfft):
        super(CNNNetwork, self).__init__()
        self.generateSpec = torchaudio.transforms.MelSpectrogram(
            n_fft=nfft, n_mels=64)

        # self.augmentSpec = nn.Sequential(torchaudio.transforms.FrequencyMasking(
        #     30, True), torchaudio.transforms.TimeMasking(20, True))

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drp = nn.Dropout2d(float(config['model']['dropout']))
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn1 = nn.BatchNorm1d(120)
        self.bn2 = nn.BatchNorm1d(24)
        self.fc1 = nn.LazyLinear(out_features=120, bias=False)
        # self.fc1 = nn.Linear(1152, 120, bias=False)
        self.fc2 = nn.Linear(120, 24, bias=False)
        self.fc3 = nn.Linear(24, 3, bias=False)

    def __normaliseSpec(self, x):
        x/=x.max()
        x = torch.clamp(x, min=1e-10)
        # x = torch.clamp(x, max=x.max()*0.9)
        x = x.log10()
        x = torch.maximum(x, x.max() - 8.0)
        x = (x + 4.0) / 4.0
        return x
        # return x/(x+10*x.median()+1e-12)

    def forward(self, wav):
        wav -= wav.mean()
        # wav /= wav.max()*3
        x = self.generateSpec(wav)
        x = self.__normaliseSpec(x)

        # if self.training:
        #     x = self.augmentSpec(x)

        x = self.drp(F.elu(self.conv1(x)))
        x = self.drp(F.elu(self.conv2(x)))
        x = self.drp(F.elu(self.conv3(x)))
        x = self.drp(self.pool1(F.elu(self.conv4(x))))
        #size = torch.flatten(x).shape[0]
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.bn1(F.elu(self.fc1(x)))
        x = self.bn2(F.elu(self.fc2(x)))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    cnn = CNNNetwork(512)
    summary(cnn.cuda(), (1, 16000))
    # (1, 64, 44) is the shape of the signal which we obtain in dataset.py
