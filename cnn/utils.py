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
                        


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
    model (torch.nn.Module): A target PyTorch model to save.
    target_dir (str): A directory for saving the model to.
    model_name (str): A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example:
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
    """
    Loads a Pytorch model state dict from folder.

    Args:
        model_path (str): A directory for loading the model from.

    Returns:
        Pytorch model
    
    Example:
        pytorch_model = load_model(model_path)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model = SignatureCNN().to(device)

    cnn_model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    return cnn_model


'''def show_random_images(images, labels):
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
'''

data_transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),  # normalize to [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])