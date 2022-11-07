from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 CNN block / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=28,
                      kernel_size=4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28,
                      out_channels=64,
                      kernel_size=5,
                      stride=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5,
                      stride=3),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=256, out_features=100)
        self.linear2 = nn.Linear(in_features=100, out_features=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        logits = self.linear1(x)
        return self.linear2(logits)


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 201, 201))
    # (1, 64, 44) is the shape of the signal which we obtain in dataset.py
