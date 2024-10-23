import torch
import torch.nn as nn
import cv2
import numpy as np


class SignatureCNN(nn.Module):
    """
    A custom CNN classifier model design to the detect authenticity of signatures.
    """

    def __init__(self, input_channels: int = 1):
        """
        Initialises the CNN architecture.
        This contains the model building blocks, and a fully connected classifier head.

        Args:
            input_channels (int): Number of input channels in the image.
        """
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Added padding=1 to maintain spatial dimensions
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
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

        # Calculate the size after conv blocks
        # After two MaxPool2d layers, spatial dimensions are reduced by factor of 4
        feature_size = 90 // 4  # 56

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * feature_size * feature_size,
                      out_features=2)  # 2 classes: real vs forged
        )

    def forward(self, x):
        """
        Defines the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_channels, 224, 224].

        Returns:
            torch.Tensor: Output probabilities from the classifier head for each class.

        Examples:
        #   >>> torch.manual_seed(42)
                model_0 = SignatureCNN(input_channels=1)

                sample = torch.rand(size=([32, 1, 224, 224]))
                model_0(sample)
        """
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


