import torch
from pathlib import Path
from model_builder import SignatureCNN
from torchvision import transforms
import warnings
from file_paths import file_path
import os
import cv2

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


def get_single_image_and_transform():
    """
    This function loads the first image from the specified prediction folder, and apply transformation (resizing and conversion to a tensor).

    Returns:
        torch.Tensor: The transformed image tensor with shape (1, 3, image_size, image_size).
    """
    image_folder_name = file_path['predict_image_folder']
    img_name = os.listdir(image_folder_name)[0]
    img_path = f'{image_folder_name}/{img_name}'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_size = file_path['image_size']
    image_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
    )
    return image_transform(img).unsqueeze(dim=0)


data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x / 255.0),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
