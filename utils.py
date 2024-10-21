import os
import torch
from pathlib import Path
from model_builder import SignatureCNN
import matplotlib.pyplot as plt
import random
from torchvision import transforms
from PIL import Image
from file_paths import file_path
import numpy as np
from model_builder import SignatureCNN, GradCAM
import cv2
import warnings
warnings.filterwarnings('ignore')
                        
device = "cuda" if torch.cuda.is_available() else "cpu"


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="cnn_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model = SignatureCNN().to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    return cnn_model


def show_random_images(images, labels):
    # Function to display random images

    plt.figure(figsize=(10, 4))

    # Get a random sample of 9 indices
    random_indices = random.sample(range(len(images)), k=3)

    for i, idx in enumerate(random_indices):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[idx].reshape(224, 224), cmap='gray')
        plt.title(f"Label: {int(labels[idx])}")
        plt.axis("off")

    plt.show()


data_transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),  # normalize to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_single_image_and_transform():
    image_path = file_path['predict_image_folder']
    img = os.listdir(image_path)[0]
    img_path = f'{image_path}/{img}'
    input_image = Image.open(img_path)

    image_size = 224
    image_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            )
    return image_transform(input_image).unsqueeze(dim=0)

def image_visualization(grad_cam, input_image, predicted_class):
    heatmap = grad_cam.generate_cam(input_image, predicted_class)

    # Normalize the heatmap
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
    transformed_input_image = get_single_image_and_transform()

    model = SignatureCNN(input_channels=3)
    model.eval()
    grad_cam = GradCAM(model, model.conv_block_2)

    with torch.no_grad():
        output = model(transformed_input_image)
        predicted_class = output.argmax(dim=1).item()

    image_visualization(grad_cam=grad_cam, input_image=transformed_input_image, predicted_class=predicted_class)