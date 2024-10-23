import matplotlib.pyplot as plt
import cv2
import numpy as np
from model_builder import SignatureCNN
import torch
from utils import get_single_image_and_transform


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
        gradients = self.gradients  # Shape: [batch_size, 1, height, width]
        activations = self.activations  # Shape: [batch_size, 1, height, width]

        # Step 4: Global average pooling of the gradients (only one channel)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # Shape: [1]

        # Step 5: Weight the activations by the pooled gradients
        # Since we have only one channel, we can directly use the scalar value
        activations *= pooled_gradients.view(-1, 1, 1)  # Shape: [1, 1, 1] for correct broadcasting

        # Step 6: Average the activations to get the heatmap
        heatmap = activations.mean(dim=1).squeeze().detach().cpu().numpy()  # Shape: [height, width]

        # Step 7: Normalize the heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Avoid division by zero

        # Step 8: Resize the heatmap to the input image size
        heatmap = cv2.resize(heatmap, (input_image.shape[3], input_image.shape[2]))  # Width, Height

        return heatmap





def grad_cam_image_visualization(grad_cam, input_image, predicted_class):
    """
    Visualizes the Grad-CAM heatmap over the input image for a given predicted class.

    Args:
        grad_cam (GradCAM): An instance of GradCAM class used to generate the heatmap.
        input_image (torch.Tensor): The input image tensor to be visualized.
        predicted_class (int): The class index predicted by the model for which the Grad-CAM is computed.

    Displays:
        Matplotlib plots showing the original image, heatmap, and superimposed image.
    """
    # Generate the grad-cam heatmap
    heatmap = grad_cam.generate_cam(input_image, predicted_class)

    # Normalize the heatmap to range [0, 1]
    heatmap = np.maximum(heatmap, 0)  # Ensure no negative values
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Avoid division by zero

    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 3, 1)
    input_image_np = input_image.squeeze().cpu().numpy()  # Shape: [90, 90]
    plt.imshow(input_image_np, cmap='gray')  # Show as grayscale for single-channel
    plt.title('Original Image')
    plt.axis('off')

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Jet color map for better visibility
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    # Superimpose the heatmap on the original image
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels for superimposing
    superimposed_img = heatmap_color * 0.4 + input_image_np * 0.6

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img / 255.0)
    plt.title("Superimposed Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def predict_outcome():
    """
    Loads an image from the prediction folder, applies transformations, performs inference with the pre-trained model,
    and visualizes the Grad-CAM heatmap for the predicted class.

    The function performs the following steps:
    1. Loads and transforms an image.
    2. Loads a pre-trained model (SignatureCNN) in evaluation mode.
    3. Applies Grad-CAM to generate a heatmap for the predicted class.
    4. Visualizes the original image, heatmap, and the superimposed image.

    Returns:
        None: The function only displays visual outputs.
    """
    transformed_input_image = get_single_image_and_transform()

    model = SignatureCNN(input_channels=1)
    model.eval()
    grad_cam = GradCAM(model, model.conv_block_2)

    with torch.no_grad():
        output = model(transformed_input_image)
        predicted_class = output.argmax(dim=1).item()

    grad_cam_image_visualization(grad_cam=grad_cam, input_image=transformed_input_image,
                                 predicted_class=predicted_class)
