import torch
import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        # MobileNetV2 Encoder
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        self.encoder = mobilenet_v2.features

        # Encoder Block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_ = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_ = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)

        # Fusion Layer
        self.fusion_conv = nn.Conv2d(1280 + 256, 512, kernel_size=1, padding=0)

        # Decoder Block
        self.upconv1 = nn.ConvTranspose2d(512 + 256, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(512 + 512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Output Layer
        self.output_conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)

        # Activations and other layers
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)



    def forward(self, x):
        # Encoder Block
        x1 = self.leaky_relu(self.conv1(x))
        x2 = self.leaky_relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x4_ = self.relu(self.conv4_(x4))
        x5 = self.relu(self.conv5(x4_))
        x5_ = self.relu(self.conv5_(x5))

        # MobileNetV2 Encoder
        x_mobilenet = self.encoder(x)
        conc = torch.cat([x_mobilenet, x5_], dim=1)
        x_fusion = self.relu(self.fusion_conv(conc))
        skip_fusion = torch.cat([x_fusion, x5_], dim=1)
        x = self.dropout(self.relu(self.upconv1(skip_fusion)))
        x5 = self.dropout(x5)
        x = torch.cat([x, x5], dim=1)
        x = self.dropout(self.relu(self.upconv2(x)))
        x4_ = self.dropout(x4)
        x = torch.cat([x, x4_], dim=1)
        x = self.dropout(self.relu(self.upconv3(x)))
        x3 = self.dropout(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.dropout(self.relu(self.upconv4(x)))
        x = self.dropout(self.relu(self.upconv5(x)))
        x = self.relu(self.upconv6(x))
        x = torch.tanh(self.output_conv(x))
        return x
