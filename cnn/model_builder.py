import torch
import torch.nn as nn
import cv2
import numpy as np


class SignatureCNN(nn.Module):
    """
    A custom CNN classifier model design to the detect authenticity of signatures.
    """

    def __init__(self, input_channels: int = 3):
        """
        Initialises the CNN architecture.
        This contains the model building blocks, and a fully connected classifier head.

        Args:
            input_channels (int): Number of input channels in the image.
        """
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # Input shape:
            nn.Conv2d(in_channels=input_channels,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # Added padding=1 to maintain spatial dimensions
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
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
        feature_size = 224 // 4  # 56

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


# Generated class and docstrings from ChatGPT
class GradCAM:
    """
    Class for generating Class Activation Maps (CAM) using the Grad-CAM technique.

    Grad-CAM helps visualize the regions of the input image that contribute most to the model's predictions.
    This technique is particularly useful for understanding CNNs and debugging models.

    Attributes:
        model (torch.nn.Module): The neural network model used for inference.
        target_layer (torch.nn.Module): The layer of the model for which Grad-CAM is computed.
        gradients (torch.Tensor): Captured gradients of the target layer during the backward pass.
        activations (torch.Tensor): Captured activations of the target layer during the forward pass.
    """

    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM instance.

        Args:
            model (torch.nn.Module): The neural network model to analyze.
            target_layer (torch.nn.Module): The layer to target for Grad-CAM visualization.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward and backward hooks on the target layer to capture activations and gradients.
        """

        def forward_hook(module, input, output):
            """Captures the activations from the target layer during the forward pass."""
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            """Captures the gradients from the target layer during the backward pass."""
            self.gradients = grad_out[0]

        # Register hooks to capture activations and gradients
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        """
        Generates the Class Activation Map (CAM) for a specific class.

        Args:
            input_image (torch.Tensor): Input tensor representing the image, shape [batch_size, channels, height, width].
            target_class (int): The class index for which to generate the CAM.

        Returns:
            np.ndarray: The normalized heatmap representing the regions of interest for the target class.
        """
        # Step 1: Forward pass
        output = self.model(input_image)

        # Step 2: Backward pass to get gradients
        self.model.zero_grad()
        output[:, target_class].backward()

        # Step 3: Get the gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Step 4: Global average pooling of the gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Step 5: Weight the activations by the pooled gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Step 6: Average the activations to get the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()

        # Step 7: Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # Step 8: Resize the heatmap to the input image size
        heatmap = cv2.resize(heatmap, (input_image.shape[2], input_image.shape[3]))

        return heatmap
