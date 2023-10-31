import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(12, 12, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(12, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.deconv1(x))
        x = torch.tanh(self.deconv2(x))
        return x
