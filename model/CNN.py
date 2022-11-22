from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from configparser import ConfigParser
import torchaudio

config = ConfigParser()
config.read('config.ini')

class CNNNetwork(nn.Module):
    def normaliseSpec(self, x):
        return x/(x + x.median()+1e-12)

    def __init__(self):
        super(CNNNetwork, self).__init__()

        self.generateSpec = torchaudio.transforms.Spectrogram(n_fft=int(config['data']['n_fft']))
        self.augmentSpec = nn.Sequential(torchaudio.transforms.FrequencyMasking(65, True), torchaudio.transforms.TimeMasking(20, True))

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drp = nn.Dropout2d(float(config['model']['dropout']))
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.bn1= nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(4, stride=4)
        self.fc1 = nn.Linear(3840, 32)
        self.fc2 = nn.Linear(32, 3)

    

    def forward(self, wav):
        x = self.generateSpec(wav)
        x = self.normaliseSpec(x)
        x = self.augmentSpec(x)

        # import matplotlib.pyplot as plt
        # plt.imshow(x[0][0].cpu())
        # plt.show()
        x = self.drp(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(self.pool1(F.relu(self.conv3(x))))
        
        # x = self.drp(self.pool1(F.relu(self.conv4(x))))
        x = self.bn1(x)
        #size = torch.flatten(x).shape[0]
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        #x = x.unsqueeze_(1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1,8000))
    # (1, 64, 44) is the shape of the signal which we obtain in dataset.py
