import torch
import torch.nn as nn


class SiameseSignatureCNN(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        feature_size = 224 // 4  # 56

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.fc(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, base_cnn: nn.Module):
        super(SiameseNetwork, self).__init__()
        self.base_cnn = base_cnn 

    def forward(self, input1, input2):
        output1 = self.base_cnn(input1)
        output2 = self.base_cnn(input2)

        distance = torch.norm(output1 - output2, p=2, dim=1)
        return distance

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label: 1 if the images are of the same class (real-real or forged-forged), 0 otherwise
        loss = 0.5 * ((label) * torch.pow(distance, 2) +
                      (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss.mean()