from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

class CNNNetwork_spec(nn.Module):
    def __init__(self):
        super(CNNNetwork_spec, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drp = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(4, stride=4)
        self.fc1 = nn.Linear(3584, 32)
        self.fc2 = nn.Linear(32, 3)


    def forward(self, x):
        x = self.drp(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(self.pool1(F.relu(self.conv3(x))))
        x = self.drp(self.pool1(F.relu(self.conv4(x))))
        #size = torch.flatten(x).shape[0]
        #print(x.shape)
        x = x.view(x.shape[0], -1)
        #x = x.unsqueeze_(1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    cnn = CNNNetwork_spec()
    summary(cnn.cuda(), (1, 257, 63))