import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
from file_paths import file_path
import numpy as np
import cv2
from model_builder import SignatureCNN, GradCAM


def get_single_image_and_transform():
    """
    This function loads the first image from the specified prediction folder, and apply transformation (resizing and conversion to a tensor).
    
    Returns:
        torch.Tensor: The transformed image tensor with shape (1, 3, image_size, image_size). 
    """
    image_path = file_path['predict_image_folder']
    img = os.listdir(image_path)[0]
    img_path = f'{image_path}/{img}'
    input_image = Image.open(img_path)

    image_size = file_path['image_size']
    image_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )
    return image_transform(input_image).unsqueeze(dim=0)

def image_visualization(grad_cam, input_image, predicted_class):
    """
    Visualizes the Grad-CAM heatmap over the input image for a given predicted class.

    The function displays three subplots: the original image, the Grad-CAM heatmap, and the superimposed image 
    with the heatmap overlaid on the original image.

    Args:
        grad_cam (GradCAM): An instance of GradCAM class used to generate the heatmap.
        input_image (torch.Tensor): The input image tensor to be visualized.
        predicted_class (int): The class index predicted by the model for which the Grad-CAM is computed.

    Displays:
        Matplotlib plots showing the original image, heatmap, and superimposed image.
    """
    # generate the grad-cam heatmap
    heatmap = grad_cam.generate_cam(input_image, predicted_class)

    # Normalize the heatmap to range [0, 1]
    heatmap = np.maximum(heatmap, 0)  # Ensure no negative values
    heatmap /= np.max(heatmap)  # Normalize to range [0, 1]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='gray')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    # Superimpose the heatmap on the original image
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    input_image_np = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
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

    model = SignatureCNN(input_channels=3)
    model.eval()
    grad_cam = GradCAM(model, model.conv_block_2)

    with torch.no_grad():
        output = model(transformed_input_image)
        predicted_class = output.argmax(dim=1).item()

    image_visualization(grad_cam=grad_cam, input_image=transformed_input_image, predicted_class=predicted_class)
    
