import torch_audiomentations as aug
from torch import nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from configparser import ConfigParser
import torchaudio
import sys
import torchvision

resnet = torchvision.models.resnet18(num_classes=3)
resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)


class featurePreprocess(nn.Module):
    def __init__(self, nfft, augmentations=None, outputClasses=3):
        super(featurePreprocess, self).__init__()

        self.audioNorm = aug.PeakNormalization(p=1)
        self.augmentor = augmentations

        self.generateSpec = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000, n_fft=nfft)

    def __normaliseSpec(self, x):
        # normalisation method 1: whisper
        x = torch.clamp(x, min=1e-10).log10()
        max_val = x.reshape(
            x.shape[0], 1, -1).amax(2).view(x.shape[0], 1, 1, 1)
        x = torch.maximum(x, max_val - 8)

        # normalisation method 2: min 0, max 1
        # x = x.log10()
        # min_val = x.reshape(
        #     x.shape[0], 1, -1).amin(2).view(x.shape[0], 1, 1, 1)
        # x -= min_val
        # max_val = x.reshape(
        #     x.shape[0], 1, -1).amax(2).view(x.shape[0], 1, 1, 1)
        # x /= max_val

        # normalisation method 3: mean 0, std 1
        # x = (x - x.mean()) / x.std()

        # normalisation method 4: median
        # x = x/(x+10*x.median()+1e-12)
        return x

    def __audioNormalisation(self, wav):
        if isinstance(wav, torch.Tensor):
            print(wav.shape)
            wav = self.audioNorm.train()(wav)
            if self.augmentor is not None:
                wav = self.augmentor(wav, 8000)
        return wav

    def forward(self, wav):
        wav = self.__audioNormalisation(wav)
        x = self.generateSpec(wav)
        x = self.__normaliseSpec(x)
        return x


resnet18 = nn.Sequential(
    featurePreprocess(512),
    resnet
)

if __name__ == "__main__":
    print(resnet18[-1].fc)
    # summary(resnet18, (1, 16000), device="cpu")
    # print(resnet18(torch.randn(2, 1, 16000)).shape)
    # resnet18.eval()
    # print(resnet18(torch.randn(2, 1, 16000)).shape)
