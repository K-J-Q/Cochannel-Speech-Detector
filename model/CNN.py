from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 CNN block / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=28,
                      kernel_size=3),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=28,
                      out_channels=64,
                      kernel_size=4,
                      stride=3),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=3),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=5,
                      stride=2),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(in_features=16128, out_features=1000)
        self.linear2 = nn.Linear(in_features=1000, out_features=3, bias=False)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = nn.functional.relu(self.linear1(x))
        logits = self.linear2(logits)
        return logits


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 201, 161))
    # (1, 64, 44) is the shape of the signal which we obtain in dataset.py
