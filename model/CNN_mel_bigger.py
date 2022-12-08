import torch
import torch.nn as nn
import torch.nn.functional as F


class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()

        # add more layers
        # input channel, output channel, kernel size
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)

        # add pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # add dropout layers
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # add fully-connected layers
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        # apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # flatten the output and apply dropout
        x = x.view(-1, 4096)
        x = self.dropout2(x)

        # apply fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    cnn = NewModel(512)
    summary(cnn.cuda(), (1, 16000))
    # (1, 64, 44) is the shape of the signal which we obtain in dataset.py
